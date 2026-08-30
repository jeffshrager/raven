;;;; experiment.lisp
;;;;
;;;; Testing jig + training driver for Raven-LLM.
;;;;
;;;; Now the top of the ASDF-loaded stack. Contains:
;;;;   - loss-log utility, TRAIN, TRAIN-LOOP, DEMO
;;;;     (moved here from the old src/main.lisp)
;;;;   - TRAIN-AND-REGISTER — read spec from models/<name>.model, train,
;;;;     save checkpoint + manifest + log entry
;;;;   - RUN-EXPERIMENT — run tests against a registered model
;;;;   - TEST-JIG — end-to-end self-test
;;;;
;;;; Directory layout (relative to CWD, usually project root):
;;;;   models/       Per model NAME:
;;;;                   NAME.model     — spec (plist, user-written)
;;;;                   NAME.ravn      — checkpoint (binary; see checkpoint.lisp)
;;;;                   NAME.manifest  — post-train metadata (alist, auto-written)
;;;;                   NAME.log       — append-only event log (alist per line)
;;;;   tests/        One challenge file per group; each non-blank,
;;;;                 non-comment line is  PROMPT|EXPECTED
;;;;                 Lines starting with ";" are comments.
;;;;   experiments/  Timestamped subdirs (YYYYMMDDHHMMSS/), each contains:
;;;;                   manifest.lisp  — copy of the experiment manifest (alist)
;;;;                   results.log    — one alist per challenge, one per line
;;;;                   summary.log    — single alist with aggregate stats
;;;;
;;;; File format note:
;;;;   .MODEL files (user-written config): PLIST, e.g.
;;;;     (:corpus-path "corpus/poeall.txt" :d-model 64 :n-layers 3 :steps 5000)
;;;;   Everything else (auto-generated metadata): ALIST of (:key . value)
;;;;   pairs. Both are S-expressions — no JSON anywhere.
;;;;
;;;; How to run (from project root):
;;;;   sbcl --load raven.asd \
;;;;        --eval '(asdf:load-system :raven)' \
;;;;        --eval '(test-jig)'
;;;;
;;;; Typical usage:
;;;;   ;; write models/poe-64d-3l.model containing:
;;;;   ;;   (:corpus-path "corpus/poeall.txt" :d-model 64 :n-layers 3 :steps 5000)
;;;;   (train-and-register "poe-64d-3l")
;;;;
;;;;   ;; write experiments/eval-1.manifest containing:
;;;;   ;;   ((:model . "poe-64d-3l") (:tests . ("raven.txt")))
;;;;   (run-experiment "experiments/eval-1.manifest")

(in-package :cl-user)


;;; ------------------------------------------------------------------
;;; Config
;;; ------------------------------------------------------------------

(defparameter *models-dir*      "models/")
(defparameter *tests-dir*       "tests/")
(defparameter *experiments-dir* "experiments/")


;;; ------------------------------------------------------------------
;;; Small helpers
;;; ------------------------------------------------------------------

(defun current-timestamp ()
  "YYYYMMDDHHMMSS from local time. Uniqueness assumes ≤ 1 experiment/sec."
  (multiple-value-bind (s m h day mo year) (decode-universal-time (get-universal-time))
    (format nil "~4,'0D~2,'0D~2,'0D~2,'0D~2,'0D~2,'0D" year mo day h m s)))

(defun save-sexp (path sexp)
  "Write SEXP to PATH in a human-readable, read-safe form."
  (ensure-directories-exist path)
  (with-open-file (s path :direction :output
                          :if-exists :supersede
                          :if-does-not-exist :create)
    (with-standard-io-syntax
      (let ((*print-readably* nil)
            (*print-escape* t)
            (*print-pretty* t))
        (prin1 sexp s)
        (terpri s)))))

(defun load-sexp (path)
  (with-open-file (s path :direction :input)
    (with-standard-io-syntax
      (read s))))

(defun append-log-line (path sexp)
  "Append PRIN1 of SEXP + newline to PATH (creates the file if missing)."
  (ensure-directories-exist path)
  (with-open-file (s path :direction :output
                          :if-exists :append
                          :if-does-not-exist :create)
    (with-standard-io-syntax
      (let ((*print-readably* nil)
            (*print-escape* t))
        (prin1 sexp s)
        (terpri s)))))

(defun model-spec-path       (name) (concatenate 'string *models-dir* name ".model"))
(defun model-checkpoint-path (name) (concatenate 'string *models-dir* name ".ravn"))
(defun model-manifest-path   (name) (concatenate 'string *models-dir* name ".manifest"))
(defun model-log-path        (name) (concatenate 'string *models-dir* name ".log"))


;;; ------------------------------------------------------------------
;;; Loss-log + generic training driver (moved from old main.lisp)
;;; ------------------------------------------------------------------

(defstruct loss-log
  (sum 0.0f0 :type single-float)
  (n   0     :type fixnum))

(defun loss-log-add! (log l)
  (incf (loss-log-sum log) l)
  (incf (loss-log-n log)))

(defun loss-log-mean-and-reset! (log)
  (let ((mean (if (zerop (loss-log-n log))
                  0.0f0
                  (/ (loss-log-sum log) (loss-log-n log)))))
    (setf (loss-log-sum log) 0.0f0
          (loss-log-n log)   0)
    mean))

(defun train-loop (encoded-corpus cfg steps lr log-every)
  "Given ENCODED-CORPUS and full model config plist CFG, build a fresh
   model and run STEPS Adam steps. Returns (values PARAMS SPEC)."
  (let* ((spec    (build-gpt-lm-spec cfg))
         (params  (build-model-params spec))
         (state   (adam-alloc params))
         (fwd     (make-model-fwd spec))
         (bwd     (make-model-bwd spec))
         (context (getf cfg :max-len))
         (log     (make-loss-log)))
    (format t "  Model: d=~D H=~D L=~D T=~D V=~D  lr=~,1E~%"
            (getf cfg :d-model) (getf cfg :n-heads) (getf cfg :n-layers)
            context (getf cfg :vocab-size) lr)
    (dotimes (step steps)
      (multiple-value-bind (input target) (sample-window encoded-corpus context)
        (let ((loss (train-step! fwd bwd params state input target :lr lr)))
          (loss-log-add! log loss)
          (when (zerop (mod (1+ step) log-every))
            (format t "  step ~5D  avg-loss ~,4E~%"
                    (1+ step) (loss-log-mean-and-reset! log))))))
    (values params spec)))

(defun train (&key corpus-path
                   (context   64)
                   (d-model   64)
                   (n-heads   4)
                   (n-layers  3)
                   (d-ff      nil)         ; NIL ⇒ 4 * d-model
                   (eps       1.0f-5)
                   (lr        1.0f-3)
                   (steps     1000)
                   (log-every 50))
  "Train a GPT-style LM on the corpus at CORPUS-PATH. Returns
   (values PARAMS SPEC VOCAB). Ad-hoc REPL entry point;
   TRAIN-AND-REGISTER wraps this and reads the same knobs from
   models/NAME.model."
  (assert corpus-path () "TRAIN: :CORPUS-PATH is required.")
  (format t "~&Loading corpus from ~A ...~%" corpus-path)
  (let* ((text   (load-corpus corpus-path))
         (vocab  (build-vocab text))
         (v      (vocab-size vocab))
         (corpus (encode vocab text))
         (cfg    (list :vocab-size v :d-model d-model :n-heads n-heads
                       :n-layers n-layers :max-len context
                       :d-ff d-ff :eps eps)))
    (format t "  corpus chars=~D  vocab=~D~%" (length text) v)
    (multiple-value-bind (params spec) (train-loop corpus cfg steps lr log-every)
      (format t "~&Training done.~%")
      (values params spec vocab))))

(defun demo (&key (steps 300) (log-every 50))
  "Self-contained synthetic training run. No corpus file needed."
  (format t "~&=== Demo training on synthetic corpus ===~%")
  (let* ((text   (with-output-to-string (s)
                   (dotimes (i 200) (declare (ignore i))
                     (write-string "the quick brown fox " s))))
         (vocab  (build-vocab text))
         (v      (vocab-size vocab))
         (corpus (encode vocab text))
         (cfg    (list :vocab-size v :d-model 16 :n-heads 2
                       :n-layers 1 :max-len 16 :d-ff 32 :eps 1.0f-5)))
    (format t "  synthetic corpus chars=~D  vocab=~D~%" (length text) v)
    (train-loop corpus cfg steps 5.0f-3 log-every)
    (format t "~&Demo done.~%")))


;;; ------------------------------------------------------------------
;;; Final-loss estimation
;;; ------------------------------------------------------------------

(defun eval-loss (params spec corpus context n-samples)
  "Estimate mean per-token loss by averaging N-SAMPLES random windows."
  (let ((fwd (make-model-fwd spec))
        (total 0.0f0))
    (declare (type single-float total))
    (dotimes (i n-samples) (declare (ignore i))
      (multiple-value-bind (input target) (sample-window corpus context)
        (multiple-value-bind (logits sv) (funcall fwd params input)
          (declare (ignore sv))
          (incf total (nth-value 0 (softmax-ce-fwd logits target))))))
    (/ total (float n-samples))))


;;; ------------------------------------------------------------------
;;; Train and register (spec-file driven)
;;; ------------------------------------------------------------------

(defun train-and-register (name)
  "Read models/NAME.model (a plist), train per that spec, save
   models/NAME.{ravn,manifest} and append a :TRAINED event to
   models/NAME.log. Returns (values PARAMS SPEC VOCAB MANIFEST-ALIST).

   Supported .model plist keys (all optional except :CORPUS-PATH,
   defaults in parens):
     :corpus-path (REQUIRED) — path to training corpus text file
     :context      (64)      — sequence length (T)
     :d-model      (64)      — model width
     :n-heads      (4)
     :n-layers     (3)
     :d-ff         (4*d-model)
     :eps          (1e-5)
     :lr           (1e-3)
     :steps        (5000)
     :log-every    (100)
     :eval-samples (30)      — # random windows averaged for :FINAL-LOSS
     :description  (NIL)     — free-text note, copied into the manifest

   Overwrites any existing .ravn / .manifest for this NAME (the .log is
   append-only, so re-training history is preserved)."
  (assert (stringp name) () "train-and-register: NAME must be a string.")
  (ensure-directories-exist *models-dir*)
  (let ((spec-path (model-spec-path name)))
    (unless (probe-file spec-path)
      (error "train-and-register: spec file ~A not found. Create it first with the model plist."
             spec-path))
    (let* ((spec         (load-sexp spec-path))
           (corpus-path  (or (getf spec :corpus-path)
                             (error "train-and-register: ~A missing :CORPUS-PATH"
                                    spec-path)))
           (context      (getf spec :context 64))
           (d-model      (getf spec :d-model 64))
           (n-heads      (getf spec :n-heads 4))
           (n-layers     (getf spec :n-layers 3))
           (d-ff         (getf spec :d-ff))
           (eps          (coerce (getf spec :eps 1.0f-5) 'single-float))
           (lr           (coerce (getf spec :lr 1.0f-3) 'single-float))
           (steps        (getf spec :steps 5000))
           (log-every    (getf spec :log-every 100))
           (eval-samples (getf spec :eval-samples 30))
           (description  (getf spec :description)))
      (multiple-value-bind (params model-spec vocab)
          (train :corpus-path corpus-path :context context :d-model d-model
                 :n-heads n-heads :n-layers n-layers :d-ff d-ff :eps eps
                 :lr lr :steps steps :log-every log-every)
        (let* ((text         (load-corpus corpus-path))
               (encoded      (encode vocab text))
               (final-loss   (eval-loss params model-spec encoded context eval-samples))
               (timestamp    (current-timestamp))
               (checkpoint   (model-checkpoint-path name))
               (manifest-p   (model-manifest-path name))
               (log-path     (model-log-path name))
               (manifest-alist
                 (list (cons :name        name)
                       (cons :corpus      corpus-path)
                       (cons :vocab-size  (vocab-size vocab))
                       (cons :config      (list (cons :context  context)
                                                (cons :d-model  d-model)
                                                (cons :n-heads  n-heads)
                                                (cons :n-layers n-layers)
                                                (cons :d-ff     (or d-ff (* 4 d-model)))
                                                (cons :eps      eps)))
                       (cons :training    (list (cons :steps        steps)
                                                (cons :lr           lr)
                                                (cons :final-loss   final-loss)
                                                (cons :eval-samples eval-samples)))
                       (cons :trained-at  timestamp)
                       (cons :description description))))
          (save-checkpoint checkpoint params model-spec vocab)
          (save-sexp manifest-p manifest-alist)
          (append-log-line log-path
                           (list (cons :event      :trained)
                                 (cons :at         timestamp)
                                 (cons :steps      steps)
                                 (cons :final-loss final-loss)))
          (format t "~&Registered model ~S~%  checkpoint: ~A~%  manifest:   ~A~%  final loss: ~,4E~%"
                  name checkpoint manifest-p final-loss)
          (values params model-spec vocab manifest-alist))))))


;;; ------------------------------------------------------------------
;;; Test file parsing
;;; ------------------------------------------------------------------

(defun parse-test-file (path)
  "Return list of alists ((:line-num . N) (:prompt . S) (:expected . S)).
   Skips blank lines and lines starting with ';'. Errors on a non-comment
   line missing the '|' separator (so a malformed file fails loudly)."
  (with-open-file (s path :direction :input)
    (let ((results nil) (n 0))
      (loop for line = (read-line s nil nil) while line do
            (incf n)
            (unless (or (zerop (length line))
                        (char= (char line 0) #\;))
              (let ((pipe (position #\| line)))
                (unless pipe
                  (error "parse-test-file ~A line ~D: no '|' separator" path n))
                (push (list (cons :line-num n)
                            (cons :prompt   (subseq line 0 pipe))
                            (cons :expected (subseq line (1+ pipe))))
                      results))))
      (nreverse results))))

(defun list-test-files ()
  "All *.txt files under *tests-dir*, as full paths."
  (mapcar #'namestring
          (directory (make-pathname :name :wild :type "txt"
                                    :defaults *tests-dir*))))


;;; ------------------------------------------------------------------
;;; Scoring
;;; ------------------------------------------------------------------

(defun score-nll (params spec vocab prompt expected)
  "Mean NLL of EXPECTED given PROMPT under the model. Deterministic.
   Requires prompt+expected ≤ model max-len."
  (let* ((prompt-ids   (encode vocab prompt))
         (expected-ids (encode vocab expected))
         (n-exp        (length expected-ids))
         (full-len     (+ (length prompt-ids) n-exp))
         (max-t        (infer-max-context spec)))
    (assert (plusp (length prompt-ids)) () "score-nll: empty prompt")
    (assert (plusp n-exp)                () "score-nll: empty expected")
    (assert (<= full-len max-t) ()
            "score-nll: prompt+expected length ~D > model max-len ~D"
            full-len max-t)
    (let* ((full   (concatenate '(simple-array fixnum (*))
                                prompt-ids expected-ids))
           (fwd    (make-model-fwd spec))
           (logits (nth-value 0 (funcall fwd params full)))
           (start  (1- (length prompt-ids)))   ; logits[start] predicts expected[0]
           (v      (array-dimension logits 1))
           (nll    0.0f0))
      (declare (type single-float nll))
      (dotimes (i n-exp)
        (let* ((pos       (+ start i))
               (target-id (aref expected-ids i))
               (lse       (logsumexp-row logits pos v)))
          (incf nll (- lse (aref logits pos target-id)))))
      (/ nll (float n-exp)))))

(defun score-challenge (params spec vocab prompt expected)
  "Deterministic NLL + greedy generation + prefix / per-char scores.
   Returns an alist."
  (let* ((n-exp (length expected))
         (nll   (score-nll params spec vocab prompt expected))
         (full  (generate params spec vocab prompt :n-tokens n-exp :greedy t))
         (gen   (subseq full (length prompt)))
         (mis   (mismatch expected gen))
         (pref  (or mis n-exp))
         (correct 0))
    (dotimes (i (min (length expected) (length gen)))
      (when (char= (char expected i) (char gen i))
        (incf correct)))
    (list (cons :nll          nll)
          (cons :greedy       gen)
          (cons :prefix-match pref)
          (cons :per-char-acc (/ (float correct) (float n-exp))))))


;;; ------------------------------------------------------------------
;;; Experiment runner
;;; ------------------------------------------------------------------

(defun run-experiment (manifest-path)
  "Read the experiment manifest at MANIFEST-PATH (an alist file), run
   tests, write logs. Manifest keys:
     :model        (required) — string, base name of models/NAME.ravn
     :tests        (optional) — list of filenames in tests/, or :ALL (default)
     :description  (optional) — free text, copied into the summary
   Returns the path of the experiment directory."
  (let* ((manifest     (load-sexp manifest-path))
         (model-name   (cdr (assoc :model manifest)))
         (tests-spec   (or (cdr (assoc :tests manifest)) :all))
         (description  (cdr (assoc :description manifest)))
         (timestamp    (current-timestamp))
         (exp-dir      (concatenate 'string *experiments-dir* timestamp "/"))
         (results-path (concatenate 'string exp-dir "results.log"))
         (summary-path (concatenate 'string exp-dir "summary.log")))
    (assert model-name () "run-experiment: manifest missing :MODEL")
    (ensure-directories-exist exp-dir)
    (save-sexp (concatenate 'string exp-dir "manifest.lisp") manifest)
    (multiple-value-bind (params spec vocab)
        (load-checkpoint (model-checkpoint-path model-name))
      (let* ((test-files (if (eq tests-spec :all)
                             (list-test-files)
                             (mapcar (lambda (f)
                                       (concatenate 'string *tests-dir* f))
                                     tests-spec)))
             (by-test    nil)
             (total-nll  0.0f0) (total-pref 0) (total-acc 0.0f0)
             (n-total    0)     (n-skipped  0))
        (declare (type single-float total-nll total-acc))
        (format t "~&=== Experiment ~A ===~%  model: ~A~%  tests: ~A~%"
                timestamp model-name test-files)
        (dolist (tf test-files)
          (let ((challenges (handler-case (parse-test-file tf)
                              (error (e)
                                (format t "  Skipping ~A: ~A~%" tf e)
                                nil)))
                (t-nll 0.0f0) (t-pref 0) (t-acc 0.0f0)
                (t-n 0) (t-skipped 0))
            (declare (type single-float t-nll t-acc))
            (dolist (ch challenges)
              (let ((prompt   (cdr (assoc :prompt ch)))
                    (expected (cdr (assoc :expected ch)))
                    (line     (cdr (assoc :line-num ch))))
                (handler-case
                    (let* ((scores (score-challenge params spec vocab prompt expected))
                           (entry  (list (cons :test-file    tf)
                                         (cons :line         line)
                                         (cons :prompt       prompt)
                                         (cons :expected     expected)
                                         (cons :greedy       (cdr (assoc :greedy scores)))
                                         (cons :nll          (cdr (assoc :nll scores)))
                                         (cons :prefix-match (cdr (assoc :prefix-match scores)))
                                         (cons :per-char-acc (cdr (assoc :per-char-acc scores))))))
                      (append-log-line results-path entry)
                      (incf t-nll  (cdr (assoc :nll          scores)))
                      (incf t-pref (cdr (assoc :prefix-match scores)))
                      (incf t-acc  (cdr (assoc :per-char-acc scores)))
                      (incf t-n))
                  (error (e)
                    (append-log-line results-path
                                     (list (cons :test-file tf)
                                           (cons :line      line)
                                           (cons :prompt    prompt)
                                           (cons :expected  expected)
                                           (cons :error     (princ-to-string e))))
                    (incf t-skipped)))))
            (when (plusp t-n)
              (push (cons (file-namestring tf)
                          (list (cons :n                 t-n)
                                (cons :skipped           t-skipped)
                                (cons :mean-nll          (/ t-nll t-n))
                                (cons :mean-prefix-match (/ (float t-pref) t-n))
                                (cons :mean-per-char-acc (/ t-acc t-n))))
                    by-test))
            (incf total-nll  t-nll)
            (incf total-pref t-pref)
            (incf total-acc  t-acc)
            (incf n-total    t-n)
            (incf n-skipped  t-skipped)))
        (let ((summary
                (list (cons :experiment-timestamp timestamp)
                      (cons :model model-name)
                      (cons :description description)
                      (cons :tests-run test-files)
                      (cons :n-challenges n-total)
                      (cons :n-skipped n-skipped)
                      (cons :mean-nll          (if (plusp n-total) (/ total-nll n-total) 0.0))
                      (cons :mean-prefix-match (if (plusp n-total) (/ (float total-pref) n-total) 0.0))
                      (cons :mean-per-char-acc (if (plusp n-total) (/ total-acc n-total) 0.0))
                      (cons :by-test (nreverse by-test)))))
          (save-sexp summary-path summary)
          (append-log-line (model-log-path model-name)
                           (list (cons :event      :evaluated)
                                 (cons :at         timestamp)
                                 (cons :experiment timestamp)
                                 (cons :mean-nll   (cdr (assoc :mean-nll summary)))))
          (format t "~&  Summary: n=~D skipped=~D  mean-nll=~,4E  mean-prefix=~,2F  mean-acc=~,3F~%"
                  n-total n-skipped
                  (cdr (assoc :mean-nll summary))
                  (cdr (assoc :mean-prefix-match summary))
                  (cdr (assoc :mean-per-char-acc summary)))
          (format t "  Wrote: ~A, ~A~%" results-path summary-path)
          exp-dir)))))


;;; ------------------------------------------------------------------
;;; End-to-end self-test
;;; ------------------------------------------------------------------

(defun test-jig ()
  "End-to-end: writes a synthetic corpus, a .model spec, a test file,
   and an experiment manifest; runs TRAIN-AND-REGISTER and RUN-EXPERIMENT
   against them; verifies expected output files exist."
  (format t "~&=== test-jig: end-to-end jig sanity ===~%")
  (let* ((root (concatenate 'string "/tmp/raven-jig-" (current-timestamp) "/"))
         (corpus-path   (concatenate 'string root "corpus.txt"))
         (test-path     (concatenate 'string root "tests/simple.txt"))
         (exp-manifest  (concatenate 'string root "experiments/my-run.manifest"))
         (*models-dir*      (concatenate 'string root "models/"))
         (*tests-dir*       (concatenate 'string root "tests/"))
         (*experiments-dir* (concatenate 'string root "experiments/")))
    (ensure-directories-exist *models-dir*)
    (ensure-directories-exist *tests-dir*)
    (ensure-directories-exist *experiments-dir*)
    ;; Synthetic corpus.
    (with-open-file (s corpus-path :direction :output
                                    :if-exists :supersede
                                    :if-does-not-exist :create)
      (dotimes (i 200) (declare (ignore i))
        (write-string "the quick brown fox " s)))
    ;; Test challenges.
    (with-open-file (s test-path :direction :output
                                  :if-exists :supersede
                                  :if-does-not-exist :create)
      (write-line "; test challenges from the synthetic corpus" s)
      (write-line "the quick brown |fox" s)
      (write-line "the quick |brown" s)
      (write-line "brown fox the |quick" s))
    ;; .model spec (plist, matches user-written format).
    (save-sexp (model-spec-path "jig-test")
               (list :corpus-path  corpus-path
                     :context      16
                     :d-model      16
                     :n-heads      2
                     :n-layers     1
                     :d-ff         32
                     :steps        300
                     :log-every    100
                     :eval-samples 5
                     :description  "jig self-test"))
    ;; Train + register from just the name.
    (train-and-register "jig-test")
    ;; Experiment manifest (alist).
    (save-sexp exp-manifest
               (list (cons :model       "jig-test")
                     (cons :tests       '("simple.txt"))
                     (cons :description "jig self-test experiment")))
    ;; Run and verify.
    (let* ((exp-dir (run-experiment exp-manifest))
           (expected-files
             (list (concatenate 'string exp-dir "manifest.lisp")
                   (concatenate 'string exp-dir "results.log")
                   (concatenate 'string exp-dir "summary.log")
                   (model-spec-path       "jig-test")
                   (model-manifest-path   "jig-test")
                   (model-log-path        "jig-test")
                   (model-checkpoint-path "jig-test")))
           (missing (remove-if #'probe-file expected-files)))
      (if (null missing)
          (format t "~&All expected files present  PASS~%")
          (format t "~&Missing files: ~A  FAIL~%" missing))
      (null missing))))
