;;;; main.lisp
;;;;
;;;; Raven-LLM entry point.
;;;;
;;;; How to run (from repo root):
;;;;   sbcl --load src/main.lisp
;;;;
;;;; By default this file runs (DEMO) — a self-contained synthetic
;;;; training run on a repeating string, so you can verify the whole
;;;; pipeline (data → model → loss → optimizer) works without touching
;;;; a corpus file.
;;;;
;;;; For real training, from the REPL or by editing the bottom of this
;;;; file:
;;;;   (train :corpus-path "corpus/poeall.txt" :steps 5000)
;;;;
;;;; TRAIN returns (values params spec vocab) so you can then run
;;;; inference (once inference.lisp lands).

(in-package :cl-user)


;;; ------------------------------------------------------------------
;;; Load all modules (compile-then-load, standard SBCL pattern)
;;; ------------------------------------------------------------------

(load (compile-file "src/utilities.lisp"))
(load (compile-file "src/tensor-ops.lisp"))
(load (compile-file "src/modules.lisp"))
(load (compile-file "src/gradcheck.lisp"))
(load (compile-file "src/training.lisp"))
(load (compile-file "src/model.lisp"))
(load (compile-file "src/data.lisp"))
(load (compile-file "src/inference.lisp"))
(load (compile-file "src/checkpoint.lisp"))


;;; ------------------------------------------------------------------
;;; Small helper: running-mean loss printer
;;; ------------------------------------------------------------------
;;;
;;; Per-step loss is noisy (each step sees a different random window).
;;; We track the running mean over LOG-EVERY steps and print that — a
;;; much cleaner training signal.

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


;;; ------------------------------------------------------------------
;;; Generic training driver
;;; ------------------------------------------------------------------
;;;
;;; Given already-built VOCAB and ENCODED-CORPUS, build a model to CFG
;;; and train STEPS steps of Adam. Kept separate from TRAIN so DEMO can
;;; share the same loop without duplicating it.

(defun train-loop (encoded-corpus cfg steps lr log-every)
  "Build a model from CFG and train STEPS steps. Returns the trained
   params plist along with SPEC (so caller can use MAKE-MODEL-FWD etc.
   afterwards)."
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


;;; ------------------------------------------------------------------
;;; Real training on a corpus file
;;; ------------------------------------------------------------------

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
   (values PARAMS SPEC VOCAB) so the caller can save or run inference."
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


;;; ------------------------------------------------------------------
;;; Self-contained demo (no corpus file required)
;;; ------------------------------------------------------------------
;;;
;;; A repeating short phrase is trivial to model, so we should see loss
;;; drop sharply within a few hundred steps. Purpose: prove the full
;;; end-to-end pipeline works after `sbcl --load src/main.lisp`.

(defun demo (&key (steps 300) (log-every 50))
  (format t "~&=== Demo training on synthetic corpus ===~%")
  (let* ((text   (with-output-to-string (s)
                   (dotimes (_ 200) (write-string "the quick brown fox " s))))
         (vocab  (build-vocab text))
         (v      (vocab-size vocab))
         (corpus (encode vocab text))
         (cfg    (list :vocab-size v :d-model 16 :n-heads 2
                       :n-layers 1 :max-len 16 :d-ff 32 :eps 1.0f-5)))
    (format t "  synthetic corpus chars=~D  vocab=~D~%" (length text) v)
    (train-loop corpus cfg steps 5.0f-3 log-every)
    (format t "~&Demo done.~%")))


;;; ------------------------------------------------------------------
;;; Top-level call
;;; ------------------------------------------------------------------
;;; Edit to call TRAIN with your corpus path when you're ready for real.

(demo)
