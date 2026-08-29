;;;; model.lisp
;;;;
;;;; Model composition for Raven-LLM.
;;;;
;;;; A model is just a sequence of module instances (embedding →
;;;; positional → N × transformer-block → final norm → unembedding).
;;;; This file provides the plumbing to allocate params for such a
;;;; sequence, thread a tensor forward through it, and thread a gradient
;;;; back through it — nothing more. The chain-rule composition lives
;;;; here and only here, so per-module code stays local to each module.
;;;;
;;;; How to run (from repo root in an SBCL REPL):
;;;;   (load (compile-file "src/utilities.lisp"))
;;;;   (load (compile-file "src/tensor-ops.lisp"))
;;;;   (load (compile-file "src/modules.lisp"))
;;;;   (load (compile-file "src/gradcheck.lisp"))
;;;;   (load (compile-file "src/training.lisp"))
;;;;   (load (compile-file "src/model.lisp"))
;;;;   (test-overfit-single-sequence)   ; end-to-end sanity
;;;;
;;;; This file is a library, not a script — no top-level side effects at load.
;;;;
;;;; Exposes:
;;;;   MODEL SPEC representation           — see comment block below
;;;;   build-model-params (spec)           — allocate a fresh param plist
;;;;   model-fwd (params spec input)       — thread forward, collect saved
;;;;   model-bwd (params spec saved grad)  — thread backward in reverse
;;;;   make-model-fwd / make-model-bwd     — closures for train-step!
;;;;   build-gpt-lm-spec (config)          — sample spec builder
;;;;   test-overfit-single-sequence         — end-to-end sanity check

(in-package :cl-user)


;;; ------------------------------------------------------------------
;;; Spec representation
;;; ------------------------------------------------------------------
;;;
;;; A SPEC is a list of triples:
;;;   ((NAME TYPE CONFIG) ...)
;;; where
;;;   NAME    is a keyword uniquely identifying this instance in the model
;;;           (e.g. :emb, :pos, :blk-0, :blk-1, :final-norm, :unemb).
;;;           NOTE: avoid :PE here — that key is special to the walker in
;;;           training.lisp (it skips :PE tables). Use :POS or similar.
;;;   TYPE    is a module type keyword registered in modules.lisp
;;;           (:embedding, :positional-sinusoidal, :rmsnorm,
;;;            :transformer-block, :unembedding, ...).
;;;   CONFIG  is the config plist that TYPE's allocator understands.
;;;
;;; PARAMS is a plist keyed by the same NAMEs, values are that module's
;;; own params plist (as returned by its allocator).
;;;
;;; SAVED is a plist keyed by the same NAMEs, values are that module's
;;; own saved plist (as returned by its forward).
;;;
;;; GRADS is a plist keyed by the same NAMEs, values are that module's
;;; own grads plist (as returned by its backward, position 2) — or NIL
;;; if the module has no trainable params. NIL entries are transparent
;;; to collect-tensors, so Adam works without special-casing.


;;; ------------------------------------------------------------------
;;; Allocation, forward, backward
;;; ------------------------------------------------------------------

(defun build-model-params (spec)
  "Allocate a fresh params plist for SPEC by calling each module's
   allocator in turn. Params are returned in spec order."
  (loop for (name type config) in spec
        collect name
        collect (module-alloc type config)))

(defun model-fwd (params spec input)
  "Run the model forward.
     PARAMS : plist of per-module params (keyed by module NAME).
     SPEC   : list of (NAME TYPE CONFIG) triples.
     INPUT  : whatever the FIRST module expects (typically token ids,
              a (simple-array fixnum (T)) for an embedding start).
   Returns (values OUTPUT SAVED) where SAVED is a plist keyed by NAME,
   containing each module's per-call saved state for backward."
  (let ((current input)
        (saved-list '()))
    (dolist (triple spec)
      (destructuring-bind (name type config) triple
        (declare (ignore config))
        (multiple-value-bind (out saved)
            (module-forward type (getf params name) current nil)
          (push name saved-list)
          (push saved saved-list)
          (setf current out))))
    (values current (nreverse saved-list))))

(defun model-bwd (params spec saved grad-out)
  "Run the model backward.
     GRAD-OUT : gradient w.r.t. the model's OUTPUT (e.g. grad-logits
                from softmax-CE backward).
   Returns (values GRAD-INPUT GRADS) where GRAD-INPUT is w.r.t. the
   model's INPUT (typically NIL for LMs, since inputs are token ids)
   and GRADS is a plist keyed by NAME containing each module's grads."
  (let ((current-grad grad-out)
        (grads-list '()))
    ;; Walk spec in reverse so we compose in the right order.
    ;; Push order: value first, then key. Because PUSH prepends, doing
    ;; (push val) (push key) leaves (key val ...) at the head — so after
    ;; the final iteration (which processes the FIRST spec module) that
    ;; module's (:name grads) ends up at position 0. The overall list
    ;; reads as a spec-order plist, matching PARAMS's key order, which
    ;; is what COLLECT-TENSORS relies on for its parallel walk in Adam.
    (dolist (triple (reverse spec))
      (destructuring-bind (name type config) triple
        (declare (ignore config))
        (multiple-value-bind (grad-in grads)
            (module-backward type
                             (getf params name)
                             (getf saved name)
                             current-grad)
          (push grads grads-list)
          (push name grads-list)
          (setf current-grad grad-in))))
    (values current-grad grads-list)))


;;; ------------------------------------------------------------------
;;; Closures for train-step!
;;; ------------------------------------------------------------------
;;;
;;; train-step! wants MODEL-FWD and MODEL-BWD to be callables of two
;;; specific shapes; the SPEC needs to be closed over.

(defun make-model-fwd (spec)
  "Return a closure (LAMBDA (PARAMS INPUT) -> (values OUTPUT SAVED))
   suitable for passing as MODEL-FWD to TRAIN-STEP!."
  (lambda (params input) (model-fwd params spec input)))

(defun make-model-bwd (spec)
  "Return a closure (LAMBDA (PARAMS SAVED GRAD) -> (values GI GRADS))
   suitable for passing as MODEL-BWD to TRAIN-STEP!."
  (lambda (params saved grad) (model-bwd params spec saved grad)))


;;; ------------------------------------------------------------------
;;; Sample spec builder — a GPT-style LM
;;; ------------------------------------------------------------------
;;;
;;; The architecture: embedding → sinusoidal PE → N × transformer-block
;;; (pre-norm, RMSNorm, GELU-FFN, no attention biases) → final RMSNorm
;;; → unembedding.
;;;
;;; Sweep-friendly: just call with a different CONFIG plist. Add other
;;; builders (e.g. build-gpt-lm-spec-rope, or a builder that swaps the
;;; norm) alongside as the project grows.

(defun build-gpt-lm-spec (config)
  "Return a spec for a standard GPT-style LM.
   Required config keys: :vocab-size :d-model :n-heads :n-layers :max-len
   Optional: :d-ff (default 4*D-MODEL), :eps (default 1e-5)."
  (let* ((v       (getf config :vocab-size))
         (d       (getf config :d-model))
         (h       (getf config :n-heads))
         (l       (getf config :n-layers))
         (max-len (getf config :max-len))
         (d-ff    (or (getf config :d-ff) (* 4 d)))
         (eps     (or (getf config :eps) 1.0f-5))
         (blocks  (loop for i below l collect
                        (list (intern (format nil "BLK-~D" i) :keyword)
                              :transformer-block
                              (list :d-model d :n-heads h
                                    :d-ff d-ff :eps eps)))))
    (append
     (list (list :emb   :embedding
                 (list :vocab-size v :d-model d))
           (list :pos   :positional-sinusoidal
                 (list :d-model d :max-len max-len)))
     blocks
     (list (list :final-norm :rmsnorm
                 (list :d-model d :eps eps))
           (list :unemb :unembedding
                 (list :d-model d :vocab-size v))))))


;;; ------------------------------------------------------------------
;;; End-to-end sanity test: overfit a single sequence
;;; ------------------------------------------------------------------
;;;
;;; The classic "does the training pipeline work?" test: pick one input
;;; sequence, one target sequence, and Adam-step on JUST that pair many
;;; times. Any correct autograd + optimizer combo will drive the loss to
;;; near zero (the model memorises the pair). If loss doesn't drop or
;;; stalls high, something's wired wrong (bad grad routing, param/grad
;;; structure mismatch, LR too small/large for the config).
;;;
;;; With V=8, D=8, H=2, L=1, T=4 this typically reaches loss < 0.05 in
;;; well under 500 steps.

(defun test-overfit-single-sequence (&key (steps 400) (lr 1.0f-2)
                                          (target-loss 0.05f0)
                                          (verbose nil))
  "Fit a tiny GPT-style model to a single (input, target) pair. PASS iff
   loss drops below TARGET-LOSS within STEPS Adam updates."
  (format t "~&Test: overfit single sequence (steps=~D lr=~,1E)~%" steps lr)
  (let* ((cfg    '(:vocab-size 8 :d-model 8 :n-heads 2
                   :n-layers 1 :max-len 4))
         (spec   (build-gpt-lm-spec cfg))
         (params (build-model-params spec))
         (state  (adam-alloc params))
         (fwd    (make-model-fwd spec))
         (bwd    (make-model-bwd spec))
         (input  (make-array 4 :element-type 'fixnum
                               :initial-contents '(0 1 2 3)))
         (target (make-array 4 :element-type 'fixnum
                               :initial-contents '(1 2 3 0)))
         (initial-loss nil)
         (final-loss   nil))
    (dotimes (i steps)
      (let ((loss (train-step! fwd bwd params state input target :lr lr)))
        (when (zerop i) (setf initial-loss loss))
        (when (= i (1- steps)) (setf final-loss loss))
        (when (and verbose (zerop (mod i 50)))
          (format t "  step ~4D  loss ~,4E~%" i loss))))
    (let ((pass (< final-loss target-loss)))
      (format t "  loss init=~,4E final=~,4E  ~A~%"
              initial-loss final-loss (if pass "PASS" "FAIL"))
      pass)))
