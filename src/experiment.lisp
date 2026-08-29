;;;; experiment.lisp
;;;;
;;;; Testing jig for Raven-LLM.
;;;;
;;;; Directory layout (relative to CWD, usually project root):
;;;;   models/       trained checkpoints, one triple per model:
;;;;                   NAME.ravn      — binary checkpoint (see checkpoint.lisp)
;;;;                   NAME.manifest  — S-expression alist of training metadata
;;;;                   NAME.log       — append-only S-expression event log
;;;;   tests/        one challenge file per group (e.g. raven.txt, cask.txt).
;;;;                 Each non-blank, non-comment line is a challenge:
;;;;                   PROMPT|EXPECTED
;;;;                 Lines starting with ";" are comments. Blank lines ignored.
;;;;   experiments/  timestamped subdirs (YYYYMMDDHHMMSS/), each contains:
;;;;                   manifest.lisp  — copy of the experiment manifest
;;;;                   results.log    — one alist per challenge, one per line
;;;;                   summary.log    — single alist with aggregate stats
;;;;
;;;; All persisted metadata is S-expression alists ((:key . value) ...).
;;;; No JSON anywhere.
;;;;
;;;; How to run (from repo root in an SBCL REPL, after loading main.lisp
;;;; and checkpoint.lisp):
;;;;   (load (compile-file "src/experiment.lisp"))
;;;;   (test-jig)
;;;;
;;;; Typical usage:
;;;;   (train-and-register :name "poe-64d-3l"
;;;;                       :corpus-path "corpus/poeall.txt"
;;;;                       :steps 5000)
;;;;   ;; write experiments/my-run.manifest — an alist with :MODEL and :TESTS
;;;;   (run-experiment "experiments/my-run.manifest")

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
  "Write SEXP to PATH in a human-readable, read-safe form. Strings are
   escaped (so they read back exactly) but their SIMPLE-BASE-STRING type
   isn't preserved — we don't need it, and *PRINT-READABLY* T would
   otherwise clutter files with '#A((N) BASE-CHAR . \"...\")' wrappers."
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
  "Append PRIN1 of SEXP + newline to PATH (creates the file if missing).
   Same *PRINT-READABLY* NIL convention as SAVE-SEXP — human-readable
   strings, still reads back correctly."
  (ensure-directories-exist path)
  (with-open-file (s path :direction :output
                          :if-exists :append
                          :if-does-not-exist :create)
    (with-standard-io-syntax
      (let ((*print-readably* nil)
            (*print-escape* t))
        (prin1 sexp s)
        (terpri s)))))

(defun model-checkpoint-path (name) (concatenate 'string *models-dir* name ".ravn"))
(defun model-manifest-path   (name) (concatenate 'string *models-dir* name ".manifest"))
(defun model-log-path        (name) (concatenate 'string *models-dir* name ".log"))


;;; ------------------------------------------------------------------
;;; Loss estimation
;;; ------------------------------------------------------------------

(defun eval-loss (params spec corpus context n-samples)
  "Estimate mean per-token loss by sampling N-SAMPLES random windows.
   Used to record a stable 'final loss' number in the model manifest."
  (let ((fwd (make-model-fwd spec))
        (total 0.0f0))
    (declare (type single-float total))
    (dotimes (i n-samples)
      (declare (ignore i))
      (multiple-value-bind (input target) (sample-window corpus context)
        (multiple-value-bind (logits sv) (funcall fwd params input)
          (declare (ignore sv))
          (incf total (nth-value 0 (softmax-ce-fwd logits target))))))
    (/ total (float n-samples))))


;;; ------------------------------------------------------------------
;;; Train and register
;;; ------------------------------------------------------------------

(defun train-and-register (&key name corpus-path
                                (context 64) (d-model 64) (n-heads 4)
                                (n-layers 3) (d-ff nil) (eps 1.0f-5)
                                (lr 1.0f-3) (steps 5000) (log-every 100)
                                (eval-samples 30)
                                (description nil))
  "Train a fresh model on CORPUS-PATH, save checkpoint + manifest under
   models/NAME.{ravn,manifest}, and append a :TRAINED event to
   models/NAME.log. Returns (values PARAMS SPEC VOCAB MANIFEST-ALIST)."
  (assert name         () "train-and-register: :NAME required")
  (assert corpus-path  () "train-and-register: :CORPUS-PATH required")
  (ensure-directories-exist *models-dir*)
  (multiple-value-bind (params spec vocab)
      (train :corpus-path corpus-path :context context :d-model d-model
             :n-heads n-heads :n-layers n-layers :d-ff d-ff :eps eps
             :lr lr :steps steps :log-every log-every)
    (let* ((text        (load-corpus corpus-path))
           (encoded     (encode vocab text))
           (final-loss  (eval-loss params spec encoded context eval-samples))
           (timestamp   (current-timestamp))
           (checkpoint  (model-checkpoint-path name))
           (manifest    (model-manifest-path name))
           (log-path    (model-log-path name))
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
      (save-checkpoint checkpoint params spec vocab)
      (save-sexp manifest manifest-alist)
      (append-log-line log-path
                       (list (cons :event      :trained)
                             (cons :at         timestamp)
                             (cons :steps      steps)
                             (cons :final-loss final-loss)))
      (format t "~&Registered model ~S~%  checkpoint: ~A~%  manifest:   ~A~%  final loss: ~,4E~%"
              name checkpoint manifest final-loss)
      (values params spec vocab manifest-alist))))


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
  "Mean NLL of EXPECTED given PROMPT under the model. Deterministic —
   doesn't sample. Assumes prompt+expected fits within model's max-len
   (a longer eval would need to slide the context window)."
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
  "Combine deterministic NLL with a greedy generation and its
   character-level scores. Returns an alist:
     (:nll . <mean nll of expected>)
     (:greedy . <string, length = length(expected)>)
     (:prefix-match . <how many leading chars agree, 0..len(expected)>)
     (:per-char-acc . <fraction of matching chars>)"
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
  "Read the experiment manifest at MANIFEST-PATH, run tests, write logs.
   Manifest keys:
     :model         (required) — string, base name of models/NAME.ravn
     :tests         (optional) — list of filenames (in tests/) or :ALL (default)
     :description   (optional) — free text, copied into the summary
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
  "Self-contained: creates a temp corpus, trains + registers a tiny
   model, writes a synthetic test file + experiment manifest, runs the
   experiment, verifies all expected output files exist."
  (format t "~&=== test-jig: end-to-end jig sanity ===~%")
  (let* ((root (concatenate 'string "/tmp/raven-jig-" (current-timestamp) "/"))
         (corpus-path   (concatenate 'string root "corpus.txt"))
         (test-path     (concatenate 'string root "tests/simple.txt"))
         (manifest-path (concatenate 'string root "experiments/my-run.manifest"))
         (*models-dir*      (concatenate 'string root "models/"))
         (*tests-dir*       (concatenate 'string root "tests/"))
         (*experiments-dir* (concatenate 'string root "experiments/")))
    (ensure-directories-exist *models-dir*)
    (ensure-directories-exist *tests-dir*)
    (ensure-directories-exist *experiments-dir*)
    ;; Synthetic corpus (repeating so model can learn it).
    (with-open-file (s corpus-path :direction :output
                                    :if-exists :supersede
                                    :if-does-not-exist :create)
      (dotimes (i 200) (declare (ignore i))
        (write-string "the quick brown fox " s)))
    ;; Synthetic test file.
    (with-open-file (s test-path :direction :output
                                  :if-exists :supersede
                                  :if-does-not-exist :create)
      (write-line "; test challenges from the synthetic corpus" s)
      (write-line "the quick brown |fox" s)
      (write-line "the quick |brown" s)
      (write-line "brown fox the |quick" s))
    ;; Train + register a tiny model.
    (train-and-register :name "jig-test"
                        :corpus-path corpus-path
                        :context 16 :d-model 16 :n-heads 2 :n-layers 1
                        :d-ff 32 :steps 300 :log-every 100 :eval-samples 5
                        :description "jig self-test")
    ;; Experiment manifest.
    (save-sexp manifest-path
               (list (cons :model       "jig-test")
                     (cons :tests       '("simple.txt"))
                     (cons :description "jig self-test experiment")))
    ;; Run.
    (let* ((exp-dir (run-experiment manifest-path))
           (expected-files
             (list (concatenate 'string exp-dir "manifest.lisp")
                   (concatenate 'string exp-dir "results.log")
                   (concatenate 'string exp-dir "summary.log")
                   (model-manifest-path   "jig-test")
                   (model-log-path        "jig-test")
                   (model-checkpoint-path "jig-test")))
           (missing (remove-if #'probe-file expected-files)))
      (if (null missing)
          (format t "~&All expected files present  PASS~%")
          (format t "~&Missing files: ~A  FAIL~%" missing))
      (null missing))))
