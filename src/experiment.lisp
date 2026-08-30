;;;; experiment.lisp
;;;;
;;;; Testing jig + training driver for Raven-LLM.
;;;;
;;;; Now the top of the ASDF-loaded stack. Contains:
;;;;   - loss-log utility, TRAIN, TRAIN-LOOP, DEMO
;;;;   - Model registry: ENSURE-MODEL looks a training-spec plist up in
;;;;     models/directory.lisp and reuses the matching :COMPLETED model
;;;;     if one already exists (exact match on canonicalized spec), else
;;;;     trains a fresh one and registers it under a timestamped id.
;;;;     Models are no longer user-named. An entry starts :INCOMPLETE
;;;;     before training begins so a crash mid-run leaves an honest
;;;;     trace rather than an orphaned, unregistered checkpoint file;
;;;;     it's updated in place to :COMPLETED or :FAILED afterward.
;;;;   - EXPAND-SCANS lets an experiment manifest sweep any number of
;;;;     training-spec keys via (:scan v1 v2 ...), cartesian-expanded.
;;;;   - RUN-EXPERIMENT reads experiments/NAME/manifest.lisp, resolves
;;;;     its :model (direct id, plain spec, or scanning spec) to one or
;;;;     more concrete models via ENSURE-MODEL, and runs the test suite
;;;;     against each, one freshly timestamped subdirectory per model.
;;;;   - TEST-JIG — end-to-end self-test (sweep, reuse, and direct-id
;;;;     lookup, all in one synthetic run)
;;;;
;;;; Directory layout (relative to CWD, usually project root):
;;;;   models/
;;;;     directory.lisp   — registry: list of alist entries, one per
;;;;                        model attempt — (:id :spec :checkpoint
;;;;                        :status :started-at ...), :status one of
;;;;                        :INCOMPLETE (training in progress or the
;;;;                        process died before finishing), :COMPLETED
;;;;                        (adds :final-loss :eval-samples
;;;;                        :completed-at), :FAILED (adds :error, the
;;;;                        condition TRAIN signaled). Read/written whole.
;;;;     build.log        — append-only human-readable audit trail
;;;;                        (one alist per line): :TRAINING-STARTED,
;;;;                        :TRAINED, :TRAINING-FAILED, :EVALUATED.
;;;;     <id>.ravn         — checkpoint for model <id> (see checkpoint.lisp)
;;;;   tests/        One challenge file per group; each non-blank,
;;;;                 non-comment line is  PROMPT|EXPECTED
;;;;                 Lines starting with ";" are comments.
;;;;   experiments/
;;;;     <name>/
;;;;       manifest.lisp   — the experiment recipe (plist, user-written)
;;;;       index.log       — append-only, one alist per run dir created
;;;;                        across all invocations of this experiment
;;;;       <timestamp>/    — one per concrete model tested by an
;;;;                        invocation (ordinal-suffixed if more than
;;;;                        one lands in the same second); NEVER named
;;;;                        from a parameter value.
;;;;         manifest.lisp    — copy of the manifest used (provenance)
;;;;         model-spec.lisp  — which model this dir's results are for:
;;;;                           (:model-id :checkpoint :spec)
;;;;         results.log      — one alist per challenge (tagged :model-id)
;;;;         summary.log      — aggregate stats + per-test breakdown
;;;;
;;;; File format note:
;;;;   User-written configs (a training-spec plist, an experiment
;;;;   manifest): PLIST, e.g.
;;;;     (:corpus-path "corpus/poeall.txt" :d-model 64 :n-layers 3 :steps 5000)
;;;;   Everything else (auto-generated metadata): ALIST of (:key . value)
;;;;   pairs. Both are S-expressions — no JSON anywhere.
;;;;
;;;; Terminology: a "training-spec" is the plist above — corpus path +
;;;; architecture + training hyperparameters. This is a DIFFERENT,
;;;; higher-level thing than the "module-spec" BUILD-GPT-LM-SPEC
;;;; returns (a list of (name type config) triples, stored inside
;;;; checkpoints) — don't confuse the two.
;;;;
;;;; How to run (from project root):
;;;;   sbcl --load raven.asd \
;;;;        --eval '(asdf:load-system :raven)' \
;;;;        --eval '(test-jig)'
;;;;
;;;; Typical usage:
;;;;   ;; write experiments/eval-1/manifest.lisp containing:
;;;;   ;;   (:model (:corpus-path "corpus/poeall.txt" :d-model 64
;;;;   ;;           :n-layers 3 :steps 5000)
;;;;   ;;    :tests ("raven.txt"))
;;;;   (run-experiment "eval-1")
;;;;
;;;;   ;; Sweeping a parameter: any number of keys may carry (:scan ...);
;;;;   ;; the full cartesian product is trained/reused and tested.
;;;;   ;;   (:model (:corpus-path "corpus/poeall.txt" :d-model 64
;;;;   ;;           :n-layers (:scan 2 3 4) :steps 5000)
;;;;   ;;    :tests :all)
;;;;   (run-experiment "depth-scan")
;;;;
;;;;   ;; To build a model ad hoc without an experiment:
;;;;   (ensure-model '(:corpus-path "corpus/poeall.txt" :d-model 64 :steps 5000))

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
  "YYYYMMDDHHMMSS from local time."
  (multiple-value-bind (s m h day mo year) (decode-universal-time (get-universal-time))
    (format nil "~4,'0D~2,'0D~2,'0D~2,'0D~2,'0D~2,'0D" year mo day h m s)))

(defun unique-timestamp (taken-p)
  "Return (CURRENT-TIMESTAMP), or that with a \"-2\", \"-3\", ... suffix
   appended — the first candidate for which (FUNCALL TAKEN-P candidate)
   is false. Shared collision-avoidance helper: model ids and
   experiment run-directory names are both timestamps, and a sweep can
   mint several of either within the same wall-clock second."
  (let ((base (current-timestamp)))
    (if (not (funcall taken-p base))
        base
        (loop for i from 2
              for candidate = (format nil "~A-~D" base i)
              unless (funcall taken-p candidate)
                return candidate))))

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

(defun read-log-lines (path)
  "Read every form appended by APPEND-LOG-LINE at PATH, in file order,
   as a list. Returns NIL if PATH doesn't exist yet."
  (if (probe-file path)
      (with-open-file (s path :direction :input)
        (with-standard-io-syntax
          (loop for form = (read s nil :eof)
                until (eq form :eof)
                collect form)))
      nil))


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

(defun train-loop (encoded-corpus cfg steps lr log-every
                   &key checkpoint-path checkpoint-every vocab
                        milestones on-milestone)
  "Given ENCODED-CORPUS and full model config plist CFG, build a fresh
   model and run STEPS Adam steps. Returns (values PARAMS SPEC).

   If CHECKPOINT-PATH is given, saves a checkpoint to it every
   CHECKPOINT-EVERY steps (overwriting each time) so a long run can be
   resumed or inspected without waiting for completion. VOCAB is
   required in that case (SAVE-CHECKPOINT needs it).

   MILESTONES, if given, is a sorted-ascending list of 1-indexed step
   counts; ON-MILESTONE — a function of (STEP PARAMS SPEC VOCAB
   ENCODED-CORPUS) — is called exactly once per milestone, in order,
   the instant that many steps have completed. Independent of
   CHECKPOINT-PATH/CHECKPOINT-EVERY (that's a single overwritten file
   for crash safety; milestones are for the caller to register
   distinct, permanent snapshots) — use either, neither, or both.
   VOCAB is required whenever ON-MILESTONE is given, same as above."
  (let* ((spec    (build-gpt-lm-spec cfg))
         (params  (build-model-params spec))
         (state   (adam-alloc params))
         (fwd     (make-model-fwd spec))
         (bwd     (make-model-bwd spec))
         (context (getf cfg :max-len))
         (log     (make-loss-log))
         (remaining-milestones milestones))
    (when (or checkpoint-path on-milestone)
      (assert vocab () "train-loop: :vocab is required when :checkpoint-path or :on-milestone is given."))
    (format t "  Model: d=~D H=~D L=~D T=~D V=~D  lr=~,1E~%"
            (getf cfg :d-model) (getf cfg :n-heads) (getf cfg :n-layers)
            context (getf cfg :vocab-size) lr)
    (dotimes (step steps)
      (multiple-value-bind (input target) (sample-window encoded-corpus context)
        (let ((loss (train-step! fwd bwd params state input target :lr lr)))
          (loss-log-add! log loss)
          (when (zerop (mod (1+ step) log-every))
            (format t "  step ~5D  avg-loss ~,4E~%"
                    (1+ step) (loss-log-mean-and-reset! log)))
          (when (and checkpoint-path checkpoint-every
                     (zerop (mod (1+ step) checkpoint-every)))
            (format t "  [checkpoint] step ~5D -> ~A~%" (1+ step) checkpoint-path)
            (save-checkpoint checkpoint-path params spec vocab))
          (when (and remaining-milestones (= (1+ step) (first remaining-milestones)))
            (pop remaining-milestones)
            (funcall on-milestone (1+ step) params spec vocab encoded-corpus)))))
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
                   (log-every 50)
                   (checkpoint-path nil)
                   (checkpoint-every nil)
                   (milestones nil)
                   (on-milestone nil))
  "Train a GPT-style LM on the corpus at CORPUS-PATH. Returns
   (values PARAMS SPEC VOCAB). Ad-hoc REPL entry point; ENSURE-MODEL
   wraps this for the registry-backed, lookup-or-build workflow.

   If CHECKPOINT-PATH is given, a checkpoint is saved there every
   CHECKPOINT-EVERY steps during training (in addition to whatever the
   caller does with the final PARAMS/SPEC/VOCAB).

   MILESTONES/ON-MILESTONE are passed straight through to TRAIN-LOOP —
   see its docstring. TRAIN-WITH-MILESTONES is the usual caller."
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
    (multiple-value-bind (params spec)
        (train-loop corpus cfg steps lr log-every
                   :checkpoint-path checkpoint-path
                   :checkpoint-every checkpoint-every
                   :vocab vocab
                   :milestones milestones
                   :on-milestone on-milestone)
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
;;; Model registry: content-addressed lookup-or-build
;;; ------------------------------------------------------------------
;;;
;;; A model is identified by its training-spec, not a user-chosen name.
;;; models/directory.lisp holds every trained model's canonical spec,
;;; timestamped id, and checkpoint path. ENSURE-MODEL is the single
;;; entry point: give it a training-spec plist, get back an existing
;;; model if one with an EQUAL canonical spec exists, else a freshly
;;; trained one.

(defun directory-path () (concatenate 'string *models-dir* "directory.lisp"))
(defun build-log-path () (concatenate 'string *models-dir* "build.log"))
(defun model-checkpoint-path-for-id (id) (concatenate 'string *models-dir* id ".ravn"))

(defun load-directory ()
  "Return the list of entries in models/directory.lisp, or NIL if no
   model has ever been trained."
  (let ((p (directory-path)))
    (if (probe-file p) (load-sexp p) nil)))

(defun save-directory (entries) (save-sexp (directory-path) entries))

(defun replace-directory-entry (id new-entry)
  "Replace the models/directory.lisp entry with :ID ID with NEW-ENTRY."
  (save-directory
   (mapcar (lambda (e) (if (string= (cdr (assoc :id e)) id) new-entry e))
           (load-directory))))

(defparameter +model-defining-keys+
  '(:corpus-path :context :d-model :n-heads :n-layers :d-ff :eps :lr :steps)
  "Keys that define what a trained model IS. Two training-specs
   describe the same model iff their CANONICALIZE-TRAINING-SPEC forms
   are EQUAL. Everything else (:checkpoint-every :log-every
   :eval-samples :description) affects only how a model is produced or
   reported, never its identity, and is excluded from comparison.")

(defun canonicalize-training-spec (spec)
  "Fill in defaults and restrict SPEC (a training-spec plist — the
   kind of thing that used to live in a .model file) to
   +MODEL-DEFINING-KEYS+ in a fixed key order, so two specs describing
   the same effective model compare EQUAL regardless of key order or
   omitted defaults. Signals an error if :corpus-path is missing.

   Known limitation: :corpus-path matches as a literal string —
   \"corpus/x.txt\" and \"./corpus/x.txt\" are treated as different
   models. Spell corpus paths the same way across manifests."
  (let* ((corpus-path (or (getf spec :corpus-path)
                          (error "canonicalize-training-spec: :corpus-path is required")))
         (d-model (getf spec :d-model 64)))
    (list :corpus-path corpus-path
          :context  (getf spec :context 64)
          :d-model  d-model
          :n-heads  (getf spec :n-heads 4)
          :n-layers (getf spec :n-layers 3)
          :d-ff     (or (getf spec :d-ff) (* 4 d-model))
          :eps      (coerce (getf spec :eps 1.0f-5) 'single-float)
          :lr       (coerce (getf spec :lr 1.0f-3) 'single-float)
          :steps    (getf spec :steps 5000))))

(defun training-spec-metadata (spec)
  "Non-defining knobs pulled out of SPEC with their defaults — how a
   model is produced/reported, not what it is. Never participates in
   registry matching.
     :checkpoint-every (1000) — mid-run checkpoint interval; NIL disables
     :log-every        (100)
     :eval-samples     (30)   — # random windows averaged for final loss
     :description      (NIL) — free-text note, copied into the entry"
  (list :checkpoint-every (getf spec :checkpoint-every 1000)
        :log-every        (getf spec :log-every 100)
        :eval-samples     (getf spec :eval-samples 30)
        :description      (getf spec :description)))

(defun find-model-by-spec (canon-spec)
  "Return the models/directory.lisp entry whose :spec is EQUAL to
   CANON-SPEC (already canonicalized) AND whose :status is :COMPLETED,
   or NIL. An :INCOMPLETE or :FAILED entry never counts as a match —
   ENSURE-MODEL always starts a fresh attempt (new id) for a spec
   whose only prior entry didn't finish successfully, rather than
   reusing a checkpoint that might be partial, stale, or missing."
  (find-if (lambda (e) (and (equal (cdr (assoc :spec e)) canon-spec)
                            (eq (cdr (assoc :status e)) :completed)))
           (load-directory)))

(defun find-model-by-id (id &key (errorp t))
  "Return the models/directory.lisp entry with :id ID. Signals an
   error if not found unless :ERRORP is NIL, in which case NIL is
   returned instead."
  (or (find id (load-directory) :key (lambda (e) (cdr (assoc :id e))) :test #'string=)
      (when errorp
        (error "find-model-by-id: no model with id ~S in ~A" id (directory-path)))))

(defun fresh-model-id ()
  "A timestamp not already used as a model id or checkpoint path."
  (unique-timestamp (lambda (cand)
                      (or (find-model-by-id cand :errorp nil)
                          (probe-file (model-checkpoint-path-for-id cand))))))

(defun make-incomplete-entry (id canon-spec checkpoint-path description)
  (list (cons :id id) (cons :spec canon-spec) (cons :checkpoint checkpoint-path)
        (cons :status :incomplete) (cons :started-at id) (cons :description description)))

(defun make-failed-entry (id canon-spec checkpoint-path description condition)
  (list (cons :id id) (cons :spec canon-spec) (cons :checkpoint checkpoint-path)
        (cons :status :failed) (cons :error (princ-to-string condition))
        (cons :started-at id) (cons :description description)))

(defun make-completed-entry (id canon-spec checkpoint-path description final-loss eval-samples
                             &key (started-at id))
  (list (cons :id id) (cons :spec canon-spec) (cons :checkpoint checkpoint-path)
        (cons :status :completed) (cons :final-loss final-loss) (cons :eval-samples eval-samples)
        (cons :started-at started-at) (cons :completed-at (current-timestamp))
        (cons :description description)))

(defun ensure-model (spec)
  "SPEC is a training-spec plist (the same shape a .model file used to
   hold). Canonicalizes it and looks it up in models/directory.lisp;
   if a :COMPLETED entry with an EQUAL canonical :spec already exists,
   reuses it without touching disk. Otherwise trains a fresh model.

   Before training starts, an :INCOMPLETE entry is appended to
   directory.lisp (checkpoint path, spec, start time) — so if the
   process dies mid-run (crash, kill, power loss), even after a
   mid-run auto-checkpoint has been written, the registry shows an
   honest :INCOMPLETE record instead of an orphaned .ravn file with no
   trace anywhere. If TRAIN signals an error, that same entry is
   updated in place to :FAILED (with the error text) and the error is
   re-signaled — callers still see the failure, but the registry now
   explains it instead of just going quiet. On success the entry is
   updated in place to :COMPLETED with the final loss. FIND-MODEL-BY-SPEC
   only ever matches :COMPLETED entries, so a spec whose only prior
   attempt is :INCOMPLETE or :FAILED always gets a fresh id on retry —
   ENSURE-MODEL never resumes into a broken entry.

   Returns (values ID CHECKPOINT-PATH ENTRY REUSED-P) on success. Never
   returns loaded tensors — callers that need PARAMS/SPEC/VOCAB call
   LOAD-CHECKPOINT on CHECKPOINT-PATH themselves. This keeps one code
   path for both branches and keeps at most one model's tensors in
   memory at a time during a sweep."
  (let* ((canon    (canonicalize-training-spec spec))
         (existing (find-model-by-spec canon)))
    (if existing
        (values (cdr (assoc :id existing)) (cdr (assoc :checkpoint existing))
                existing t)
        (let* ((meta            (training-spec-metadata spec))
               (id              (fresh-model-id))
               (checkpoint-path (model-checkpoint-path-for-id id))
               (description     (getf meta :description)))
          (save-directory (append (load-directory)
                                  (list (make-incomplete-entry id canon checkpoint-path description))))
          (append-log-line (build-log-path)
                           (list (cons :event :training-started) (cons :id id)
                                 (cons :at id) (cons :spec canon)))
          (handler-case
              (multiple-value-bind (params module-spec vocab)
                  (train :corpus-path (getf canon :corpus-path)
                         :context     (getf canon :context)
                         :d-model     (getf canon :d-model)
                         :n-heads     (getf canon :n-heads)
                         :n-layers    (getf canon :n-layers)
                         :d-ff        (getf canon :d-ff)
                         :eps         (getf canon :eps)
                         :lr          (getf canon :lr)
                         :steps       (getf canon :steps)
                         :log-every        (getf meta :log-every)
                         :checkpoint-path  checkpoint-path
                         :checkpoint-every (getf meta :checkpoint-every))
                (let* ((text       (load-corpus (getf canon :corpus-path)))
                       (encoded    (encode vocab text))
                       (final-loss (eval-loss params module-spec encoded
                                              (getf canon :context) (getf meta :eval-samples)))
                       (entry (make-completed-entry id canon checkpoint-path description
                                                    final-loss (getf meta :eval-samples))))
                  (save-checkpoint checkpoint-path params module-spec vocab)
                  (replace-directory-entry id entry)
                  (append-log-line (build-log-path)
                                   (list (cons :event :trained) (cons :id id) (cons :at id)
                                         (cons :spec canon) (cons :final-loss final-loss)))
                  (format t "~&Trained model ~A~%  checkpoint: ~A~%  final loss: ~,4E~%"
                          id checkpoint-path final-loss)
                  (values id checkpoint-path entry nil)))
            (error (e)
              (replace-directory-entry id (make-failed-entry id canon checkpoint-path description e))
              (append-log-line (build-log-path)
                               (list (cons :event :training-failed) (cons :id id) (cons :at id)
                                     (cons :spec canon) (cons :error (princ-to-string e))))
              (format t "~&Training FAILED for model ~A: ~A~%" id e)
              (error e)))))))


;;; ------------------------------------------------------------------
;;; Training-length scans: (:scan ...) on :steps
;;; ------------------------------------------------------------------
;;;
;;; A key insight: if several requested models share every defining
;;; key EXCEPT :steps, they aren't independent — a single continuous
;;; run for the largest step count passes through exactly the state a
;;; standalone run for each smaller count would have stopped at. So
;;; instead of training sum(steps) total across separate runs, we can
;;; train max(missing-steps) ONCE and register a full, independently
;;; :COMPLETED model at every requested step count along the way.
;;; TRAIN-LOOP's MILESTONES/ON-MILESTONE hook is what makes this
;;; possible without duplicating any training-loop code here.

(defun train-with-milestones (spec milestone-steps)
  "Train ONE model on SPEC for (the largest value in MILESTONE-STEPS)
   steps, registering a fresh, independently :COMPLETED
   models/directory.lisp entry at EVERY step count in MILESTONE-STEPS
   along the way — not just the end. SPEC's own :STEPS (if any) is
   ignored; MILESTONE-STEPS alone determines how far training runs and
   which step counts get registered.

   Crash safety mirrors ENSURE-MODEL: an :INCOMPLETE placeholder for
   the LARGEST step count is registered before training starts, so a
   crash before it's reached leaves an honest trace (or :FAILED, if
   TRAIN signals an error — re-signaled so the caller still sees the
   failure). Smaller milestones already reached before a crash are
   genuinely :COMPLETED — they really are valid, independent, usable
   trained models the instant their snapshot is taken.

   Returns the id of the largest-step-count model."
  (let* ((sorted         (sort (copy-list (remove-duplicates milestone-steps)) #'<))
         (max-n          (car (last sorted)))
         (canon          (canonicalize-training-spec (append (list :steps max-n) spec)))
         (meta           (training-spec-metadata spec))
         (description    (getf meta :description))
         (max-id         (fresh-model-id))
         (max-checkpoint (model-checkpoint-path-for-id max-id)))
    (save-directory (append (load-directory)
                            (list (make-incomplete-entry max-id canon max-checkpoint description))))
    (append-log-line (build-log-path)
                     (list (cons :event :training-started) (cons :id max-id)
                           (cons :at max-id) (cons :spec canon)))
    (handler-case
        (progn
          (train :corpus-path (getf canon :corpus-path)
                 :context     (getf canon :context)
                 :d-model     (getf canon :d-model)
                 :n-heads     (getf canon :n-heads)
                 :n-layers    (getf canon :n-layers)
                 :d-ff        (getf canon :d-ff)
                 :eps         (getf canon :eps)
                 :lr          (getf canon :lr)
                 :steps       max-n
                 :log-every   (getf meta :log-every)
                 :milestones  sorted
                 :on-milestone
                 (lambda (n params module-spec vocab encoded-corpus)
                   (let* ((is-final        (= n max-n))
                          (id              (if is-final max-id (fresh-model-id)))
                          (checkpoint-path (if is-final max-checkpoint (model-checkpoint-path-for-id id)))
                          (milestone-canon (canonicalize-training-spec (append (list :steps n) spec)))
                          (final-loss (eval-loss params module-spec encoded-corpus
                                                 (getf canon :context) (getf meta :eval-samples)))
                          (entry (make-completed-entry id milestone-canon checkpoint-path description
                                                       final-loss (getf meta :eval-samples)
                                                       :started-at (if is-final max-id id))))
                     (save-checkpoint checkpoint-path params module-spec vocab)
                     (if is-final
                         (replace-directory-entry max-id entry)
                         (save-directory (append (load-directory) (list entry))))
                     (append-log-line (build-log-path)
                                      (list (cons :event :trained) (cons :id id) (cons :at id)
                                            (cons :spec milestone-canon) (cons :final-loss final-loss)))
                     (format t "~&  [milestone] step ~5D -> model ~A  final-loss=~,4E~%"
                             n id final-loss))))
          max-id)
      (error (e)
        (replace-directory-entry max-id (make-failed-entry max-id canon max-checkpoint description e))
        (append-log-line (build-log-path)
                         (list (cons :event :training-failed) (cons :id max-id) (cons :at max-id)
                               (cons :spec canon) (cons :error (princ-to-string e))))
        (format t "~&Training FAILED (milestone run) at id ~A: ~A~%" max-id e)
        (error e)))))

(defun ensure-model-milestones (spec steps-list)
  "Like calling ENSURE-MODEL once per value in STEPS-LIST (each with
   SPEC's :steps overridden to that value), but far cheaper when more
   than one value needs training: TRAIN-WITH-MILESTONES trains the
   largest MISSING value once and registers every missing value along
   the way, instead of training sum(missing) steps across separate
   runs. Values that already have a :COMPLETED entry are left
   untouched (reused).

   Returns a list of (ID CHECKPOINT-PATH ENTRY REUSED-P) tuples, one
   per value in STEPS-LIST, in STEPS-LIST's original order."
  (flet ((canon-for (n) (canonicalize-training-spec (append (list :steps n) spec))))
    (let* ((missing (loop for n in steps-list
                          unless (find-model-by-spec (canon-for n)) collect n)))
      (when missing
        (train-with-milestones spec (sort (copy-list missing) #'<)))
      (mapcar (lambda (n)
                (let ((e (find-model-by-spec (canon-for n))))
                  (list (cdr (assoc :id e)) (cdr (assoc :checkpoint e)) e
                        (not (member n missing)))))
              steps-list))))


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
;;; Parameter sweeps: (:scan v1 v2 ...) expansion
;;; ------------------------------------------------------------------

(defun spec-scan-p (v)
  "True if V is a (:scan v1 v2 ...) sweep directive."
  (and (consp v) (eq (first v) :scan)))

(defun spec-scan-keys (spec)
  "List of (KEY . VALUES) for every (:scan v1 v2 ...)-valued key in
   SPEC (a plist), in the order the keys first appear."
  (loop for (k v) on spec by #'cddr
        when (spec-scan-p v) collect (cons k (rest v))))

(defun expand-scans (spec)
  "Expand every (:scan v1 v2 ...) value in SPEC into the cartesian
   product of concrete plists, one per combination of scanned values —
   works for any number of scanned keys, each with any number of
   values. With no (:scan ...) values, returns (list SPEC) unchanged,
   so a non-sweep manifest goes through the same code path as a sweep
   with N=1."
  (let ((scan-keys (spec-scan-keys spec)))
    (if (null scan-keys)
        (list (copy-list spec))
        (let ((base (loop for (k v) on spec by #'cddr
                          unless (assoc k scan-keys) append (list k v))))
          (labels ((combos (keys)
                     (if (null keys)
                         (list nil)
                         (let ((k (caar keys)) (vals (cdar keys)))
                           (loop for v in vals
                                 append (mapcar (lambda (rest) (list* k v rest))
                                                (combos (rest keys))))))))
            (mapcar (lambda (combo) (append base combo)) (combos scan-keys)))))))


;;; ------------------------------------------------------------------
;;; Experiment runner
;;; ------------------------------------------------------------------

(defun experiment-manifest-path (name) (concatenate 'string *experiments-dir* name "/manifest.lisp"))
(defun experiment-dir           (name) (concatenate 'string *experiments-dir* name "/"))
(defun experiment-index-path    (name) (concatenate 'string (experiment-dir name) "index.log"))

(defun resolve-experiment-models (model-designator)
  "MODEL-DESIGNATOR is either a training-spec plist (possibly with
   :SCAN values on any number of keys) or a model id string naming an
   existing model directly. Returns a list of (ID CHECKPOINT-PATH
   ENTRY SCANNED) tuples, one per concrete model to test. SCANNED is
   an alist of just the varying key/value(s) for that concrete model
   (NIL for a direct-id designator or an unscanned plist).

   When :STEPS is one of the scanned keys, resolution goes through
   ENSURE-MODEL-MILESTONES instead of one ENSURE-MODEL call per
   combination — since every scanned :STEPS value shares the same
   architecture/corpus/lr and differs only in how long training ran,
   this trains max(:steps values) once per combination of the OTHER
   scanned keys, rather than training each value independently."
  (if (stringp model-designator)
      (let ((entry (find-model-by-id model-designator)))
        (unless (eq (cdr (assoc :status entry)) :completed)
          (format t "~&WARNING: model ~A has status ~S, not :COMPLETED — results may reflect a partial or failed training run.~%"
                  model-designator (cdr (assoc :status entry))))
        (list (list model-designator (cdr (assoc :checkpoint entry)) entry nil)))
      (let ((scan-keys (mapcar #'car (spec-scan-keys model-designator))))
        (if (member :steps scan-keys)
            (resolve-experiment-models-steps-scan model-designator scan-keys)
            (mapcar (lambda (concrete)
                      (multiple-value-bind (id path entry reused-p) (ensure-model concrete)
                        (declare (ignore reused-p))
                        (list id path entry
                              (mapcar (lambda (k) (cons k (getf concrete k))) scan-keys))))
                    (expand-scans model-designator))))))

(defun resolve-experiment-models-steps-scan (model-designator scan-keys)
  "Handle a MODEL-DESIGNATOR whose SCAN-KEYS include :STEPS. Expands
   every OTHER scanned key via EXPAND-SCANS as usual (cartesian
   product); within each resulting combination, the requested :STEPS
   values are resolved together in one shot via ENSURE-MODEL-MILESTONES."
  (let* ((steps-values  (rest (getf model-designator :steps)))
         (other-keys    (remove :steps scan-keys))
         (without-steps (loop for (k v) on model-designator by #'cddr
                              unless (eq k :steps) append (list k v)))
         (other-combos  (expand-scans without-steps)))
    (loop for combo in other-combos
          append (mapcar (lambda (result n)
                           (destructuring-bind (id path entry reused-p) result
                             (declare (ignore reused-p))
                             (list id path entry
                                   (cons (cons :steps n)
                                         (mapcar (lambda (k) (cons k (getf combo k))) other-keys)))))
                         (ensure-model-milestones combo steps-values)
                         steps-values))))

(defun run-tests-against-model (params spec vocab model-id test-files results-path)
  "Run every challenge in TEST-FILES against the loaded model, writing
   one alist per challenge to RESULTS-PATH (each tagged with
   :MODEL-ID). Returns (values BY-TEST TOTAL-NLL TOTAL-PREF TOTAL-ACC
   N-TOTAL N-SKIPPED)."
  (let ((by-test    nil)
        (total-nll  0.0f0) (total-pref 0) (total-acc 0.0f0)
        (n-total    0)     (n-skipped  0))
    (declare (type single-float total-nll total-acc))
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
                       (entry  (list (cons :model-id    model-id)
                                     (cons :test-file    tf)
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
                                 (list (cons :model-id model-id)
                                       (cons :test-file tf)
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
    (values (nreverse by-test) total-nll total-pref total-acc n-total n-skipped)))

(defun run-experiment (name)
  "Read experiments/NAME/manifest.lisp (a plist) and run it. Manifest
   keys:
     :model        (required) — a training-spec plist (see
                    CANONICALIZE-TRAINING-SPEC for keys; any value may
                    be (:scan v1 v2 ...) to sweep that parameter, and
                    any number of keys may be scanned at once — the
                    full cartesian product is trained-or-reused and
                    tested), OR a model id string to test one exact
                    existing model directly (never retrains).
     :tests        (optional) — list of filenames in tests/, or :ALL (default)
     :description  (optional) — free text, copied into each summary

   For each concrete model resolved from :MODEL, writes a fresh
   timestamped subdirectory experiments/NAME/<timestamp[-N]>/
   containing manifest.lisp (copy), model-spec.lisp (which model this
   run tested), results.log, and summary.log; also appends one line
   to experiments/NAME/index.log and an :EVALUATED line to
   models/build.log. Returns the list of run-directory paths created,
   in resolution order."
  (let* ((manifest-path    (experiment-manifest-path name))
         (manifest         (load-sexp manifest-path))
         (model-designator (getf manifest :model))
         (tests-spec       (getf manifest :tests :all))
         (description      (getf manifest :description))
         (exp-dir          (experiment-dir name))
         (index-path       (experiment-index-path name))
         (test-files (if (eq tests-spec :all)
                        (list-test-files)
                        (mapcar (lambda (f) (concatenate 'string *tests-dir* f)) tests-spec)))
         (used-run-dirs nil)
         (created-dirs  nil))
    (assert model-designator () "run-experiment: ~A missing :MODEL" manifest-path)
    (flet ((run-one (resolved)
             (destructuring-bind (model-id checkpoint entry scanned) resolved
               (let* ((ts (unique-timestamp
                           (lambda (cand)
                             (or (member cand used-run-dirs :test #'string=)
                                 (probe-file (concatenate 'string exp-dir cand "/"))))))
                      (run-dir      (concatenate 'string exp-dir ts "/"))
                      (results-path (concatenate 'string run-dir "results.log"))
                      (summary-path (concatenate 'string run-dir "summary.log")))
                 (push ts used-run-dirs)
                 (ensure-directories-exist run-dir)
                 (save-sexp (concatenate 'string run-dir "manifest.lisp") manifest)
                 (save-sexp (concatenate 'string run-dir "model-spec.lisp")
                            (list (cons :model-id model-id)
                                  (cons :checkpoint checkpoint)
                                  (cons :spec (cdr (assoc :spec entry)))))
                 (format t "~&=== Experiment ~A / ~A ===~%  model: ~A~%  tests: ~A~%"
                         name ts model-id test-files)
                 (multiple-value-bind (params module-spec vocab) (load-checkpoint checkpoint)
                   (multiple-value-bind (by-test total-nll total-pref total-acc n-total n-skipped)
                       (run-tests-against-model params module-spec vocab model-id
                                                test-files results-path)
                     (let* ((mean-nll  (if (plusp n-total) (/ total-nll n-total) 0.0))
                            (mean-pref (if (plusp n-total) (/ (float total-pref) n-total) 0.0))
                            (mean-acc  (if (plusp n-total) (/ total-acc n-total) 0.0))
                            (summary
                              (list (cons :experiment name) (cons :run ts)
                                    (cons :model-id model-id)
                                    (cons :spec (cdr (assoc :spec entry)))
                                    (cons :description description)
                                    (cons :tests-run test-files)
                                    (cons :n-challenges n-total) (cons :n-skipped n-skipped)
                                    (cons :mean-nll mean-nll)
                                    (cons :mean-prefix-match mean-pref)
                                    (cons :mean-per-char-acc mean-acc)
                                    (cons :by-test by-test))))
                       (save-sexp summary-path summary)
                       (append-log-line index-path
                                        (list (cons :run ts) (cons :model-id model-id)
                                              (cons :scanned scanned)
                                              (cons :n-challenges n-total)
                                              (cons :n-skipped n-skipped)
                                              (cons :mean-nll mean-nll)
                                              (cons :mean-prefix-match mean-pref)
                                              (cons :mean-per-char-acc mean-acc)))
                       (append-log-line (build-log-path)
                                        (list (cons :event :evaluated) (cons :id model-id)
                                              (cons :experiment name) (cons :run ts)
                                              (cons :mean-nll mean-nll)))
                       (format t "~&  Summary: n=~D skipped=~D  mean-nll=~,4E  mean-prefix=~,2F  mean-acc=~,3F~%"
                               n-total n-skipped mean-nll mean-pref mean-acc)
                       (format t "  Wrote: ~A, ~A~%" results-path summary-path)
                       (push run-dir created-dirs)))))))
             )
      (dolist (resolved (resolve-experiment-models model-designator))
        (run-one resolved)))
    (nreverse created-dirs)))


;;; ------------------------------------------------------------------
;;; End-to-end self-test
;;; ------------------------------------------------------------------

(defun test-jig ()
  "End-to-end: writes a synthetic corpus, a test file, and an
   experiment manifest that scans one training-spec parameter across
   two values; runs RUN-EXPERIMENT twice (to prove model reuse) and
   once more against a direct model id. Also exercises a :steps
   milestone scan (proving it trains once, not once per value) and
   the training-failure path (bad corpus path -> :FAILED entry).
   Checks the resulting registry and experiment-directory state at
   each step."
  (format t "~&=== test-jig: end-to-end jig sanity ===~%")
  (let* ((root        (concatenate 'string "/tmp/raven-jig-" (current-timestamp) "/"))
         (corpus-path (concatenate 'string root "corpus.txt"))
         (test-path   (concatenate 'string root "tests/simple.txt"))
         (*models-dir*      (concatenate 'string root "models/"))
         (*tests-dir*       (concatenate 'string root "tests/"))
         (*experiments-dir* (concatenate 'string root "experiments/"))
         (pass t))
    (flet ((check (label ok)
             (format t "  ~A  ~A~%" label (if ok "PASS" "FAIL"))
             (unless ok (setf pass nil))))
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
      ;; Experiment manifest: scan :n-layers over two values.
      (save-sexp (experiment-manifest-path "my-run")
                 (list :model (list :corpus-path  corpus-path
                                    :context      16
                                    :d-model      16
                                    :n-heads      2
                                    :n-layers     (list :scan 1 2)
                                    :d-ff         32
                                    :steps        300
                                    :eval-samples 5)
                       :tests '("simple.txt")
                       :description "jig self-test sweep"))
      ;; First run: 2 models get trained fresh, 2 run dirs written.
      (let ((dirs1 (run-experiment "my-run")))
        (check "first run produced 2 run dirs" (= (length dirs1) 2))
        (check "directory.lisp has 2 entries" (= (length (load-directory)) 2))
        (check "both entries are :completed"
               (every (lambda (e) (eq (cdr (assoc :status e)) :completed)) (load-directory)))
        (check "build.log has 2 :trained lines"
               (= 2 (count :trained (read-log-lines (build-log-path))
                          :key (lambda (e) (cdr (assoc :event e))))))
        (check "every run dir has all 4 files"
               (every (lambda (d)
                        (every #'probe-file
                               (mapcar (lambda (f) (concatenate 'string d f))
                                       '("manifest.lisp" "model-spec.lisp"
                                         "results.log" "summary.log"))))
                      dirs1))
        (check "index.log has 2 lines"
               (= 2 (length (read-log-lines (experiment-index-path "my-run")))))
        ;; Second run of the SAME manifest: models must be REUSED (no
        ;; retraining), but experiment run dirs are always fresh.
        (let ((dirs2 (run-experiment "my-run")))
          (check "second run produced 2 more run dirs" (= (length dirs2) 2))
          (check "no overlap between the two runs' dirs"
                 (null (intersection dirs1 dirs2 :test #'string=)))
          (check "directory.lisp still has 2 entries (models reused)"
                 (= (length (load-directory)) 2))
          (check "build.log still has only 2 :trained lines"
                 (= 2 (count :trained (read-log-lines (build-log-path))
                            :key (lambda (e) (cdr (assoc :event e))))))
          (check "index.log now has 4 lines"
                 (= 4 (length (read-log-lines (experiment-index-path "my-run"))))))
        ;; Direct-id manifest: reuse one exact model by id, no scan.
        (let* ((one-id (cdr (assoc :id (first (load-directory)))))
               (dirs-before (length (load-directory))))
          (save-sexp (experiment-manifest-path "my-run-direct")
                     (list :model one-id :tests '("simple.txt")
                           :description "direct id re-eval"))
          (let ((dirs3 (run-experiment "my-run-direct")))
            (check "direct-id run produced exactly 1 run dir" (= (length dirs3) 1))
            (check "direct-id run didn't touch directory.lisp"
                   (= (length (load-directory)) dirs-before)))))
      ;; Training-length scan: (:scan ...) on :steps should train ONCE
      ;; for the max value and register all three milestones, not
      ;; train independently three times.
      (let ((n-before        (length (load-directory)))
            (started-before  (count :training-started (read-log-lines (build-log-path))
                                    :key (lambda (e) (cdr (assoc :event e)))))
            (trained-before  (count :trained (read-log-lines (build-log-path))
                                    :key (lambda (e) (cdr (assoc :event e))))))
        (save-sexp (experiment-manifest-path "steps-scan")
                   ;; d-ff 48 is deliberately distinct from the "my-run"
                   ;; fixture above (d-ff 32) so none of these three
                   ;; specs can accidentally already exist in the
                   ;; registry — every one of them must be genuinely new.
                   (list :model (list :corpus-path  corpus-path
                                      :context      16 :d-model 16 :n-heads 2
                                      :n-layers     1 :d-ff 48
                                      :steps        (list :scan 100 200 300))
                         :tests '("simple.txt")
                         :description "jig self-test: training-length milestone scan"))
        (let ((dirs (run-experiment "steps-scan")))
          (check "steps-scan produced 3 run dirs" (= (length dirs) 3))
          (check "steps-scan added exactly 3 directory entries"
                 (= (length (load-directory)) (+ n-before 3)))
          (check "steps-scan logged exactly 1 :training-started (one continuous run)"
                 (= 1 (- (count :training-started (read-log-lines (build-log-path))
                                :key (lambda (e) (cdr (assoc :event e))))
                         started-before)))
          (check "steps-scan logged exactly 3 :trained (one per milestone)"
                 (= 3 (- (count :trained (read-log-lines (build-log-path))
                                :key (lambda (e) (cdr (assoc :event e))))
                         trained-before)))
          (check "all 3 milestone entries are :completed with the right :steps"
                 (equal '(100 200 300)
                        (sort (mapcar (lambda (d)
                                        (getf (cdr (assoc :spec (load-sexp (concatenate 'string d "model-spec.lisp"))))
                                              :steps))
                                      dirs)
                              #'<)))
          ;; Re-running the same manifest: every milestone already
          ;; :completed, so no training at all should happen this time.
          (run-experiment "steps-scan")
          (check "re-running steps-scan trains nothing new"
                 (= (length (load-directory)) (+ n-before 3)))))
      ;; Failure path: a spec whose corpus doesn't exist should error out
      ;; of ENSURE-MODEL, but leave a :FAILED (not orphaned) trace behind.
      (let ((n-before (length (load-directory)))
            (bad-spec (list :corpus-path (concatenate 'string root "no-such-corpus.txt")
                            :d-model 8 :n-heads 2 :n-layers 1 :context 8 :steps 10)))
        (let ((errored (handler-case (progn (ensure-model bad-spec) nil)
                        (error () t))))
          (check "ensure-model on a bad spec signals an error" errored))
        (let* ((dir (load-directory))
               (new-entry (find-if (lambda (e) (not (eq (cdr (assoc :status e)) :completed)))
                                   dir :from-end t)))
          (check "exactly one new directory entry was added" (= (length dir) (1+ n-before)))
          (check "the new entry is :failed with an :error message"
                 (and new-entry
                      (eq (cdr (assoc :status new-entry)) :failed)
                      (stringp (cdr (assoc :error new-entry)))))
          (check "build.log recorded :training-failed"
                 (find-if (lambda (e) (eq (cdr (assoc :event e)) :training-failed))
                          (read-log-lines (build-log-path)))))))
    (if pass
        (format t "~&All checks passed  PASS~%")
        (format t "~&Some checks failed  FAIL~%"))
    pass))
