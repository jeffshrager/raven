;;;; training.lisp
;;;;
;;;; Loss, optimizer, and training-loop composer for Raven-LLM.
;;;;
;;;; How to run (from repo root in an SBCL REPL):
;;;;   (load (compile-file "src/utilities.lisp"))
;;;;   (load (compile-file "src/tensor-ops.lisp"))
;;;;   (load (compile-file "src/modules.lisp"))
;;;;   (load (compile-file "src/gradcheck.lisp"))
;;;;   (load (compile-file "src/training.lisp"))
;;;;   (run-training-tests)   ; softmax-CE gradcheck + Adam sanity fit
;;;;
;;;; This file is a library, not a script — no top-level side effects at load.
;;;;
;;;; Exposes:
;;;;   softmax-ce-fwd / softmax-ce-bwd     fused loss (numerically stable)
;;;;   collect-tensors / params-map-tensors walkers over nested-plist params
;;;;   adam-alloc / adam-step!             optimizer (in-place update)
;;;;   train-step!                         one full forward/back/update cycle
;;;;   run-training-tests                  sanity checks

(in-package :cl-user)


;;; ------------------------------------------------------------------
;;; Fused softmax + cross-entropy loss
;;; ------------------------------------------------------------------
;;;
;;; For LM training: LOGITS Z has shape (T, V) and TARGET-IDS has shape (T,).
;;; Per-token loss is the negative log-likelihood at the target token:
;;;
;;;     L_t = -log(softmax(Z[t])[target_t])
;;;         = logsumexp(Z[t]) - Z[t, target_t]
;;;
;;; Reported loss = mean over t (per convention in LM training; makes the
;;; number equal to log-perplexity and keeps LR portable across T).
;;;
;;; Why "fused":
;;;   Computing softmax then taking log is numerically bad (large negative
;;;   logits underflow to 0, then log(0) = -inf). Doing logsumexp with the
;;;   max-subtraction trick is stable, and the gradient collapses to a very
;;;   clean form:
;;;
;;;     dL/dZ[t, k] = (1/T) * (softmax(Z[t])[k] - I[k = target_t])
;;;
;;; Backward saves the softmax probabilities and target ids from forward
;;; and reads out the gradient in one pass.

(defun logsumexp-row (logits t-idx v)
  "Numerically stable log-sum-exp of row T-IDX of LOGITS (shape (T, V)).
   Uses the standard max-subtraction trick: log(sum exp(x)) =
   max(x) + log(sum exp(x - max(x)))."
  (declare (type tensor-2d logits)
           (type fixnum t-idx v))
  (let ((mx most-negative-single-float))
    (declare (type single-float mx))
    (dotimes (j v)
      (let ((x (aref logits t-idx j)))
        (when (> x mx) (setf mx x))))
    (let ((acc 0.0f0))
      (declare (type single-float acc))
      (dotimes (j v)
        (incf acc (exp (- (aref logits t-idx j) mx))))
      (+ mx (log acc)))))

(defun softmax-ce-fwd (logits target-ids)
  "Fused softmax + cross-entropy.
     LOGITS     : tensor-2d of shape (T, V).
     TARGET-IDS : (simple-array fixnum (T)) of target token indices in [0, V).
   Returns (values LOSS SAVED):
     LOSS  : single-float, mean per-token NLL.
     SAVED : plist for backward (:probs <softmax(T,V)> :target-ids <ids>)."
  (declare (type tensor-2d logits)
           (type (simple-array fixnum (*)) target-ids))
  (let* ((t-len (array-dimension logits 0))
         (v     (array-dimension logits 1))
         (probs (tensor-like logits))
         (total 0.0f0))
    (declare (type fixnum t-len v)
             (type tensor-2d probs)
             (type single-float total))
    (dotimes (tt t-len)
      (let* ((lse       (logsumexp-row logits tt v))
             (tgt       (aref target-ids tt))
             (tgt-logit (aref logits tt tgt)))
        (declare (type single-float lse tgt-logit)
                 (type fixnum tgt))
        (incf total (- lse tgt-logit))
        ;; Row of softmax = exp(logits - lse) — same shift as inside lse.
        (dotimes (j v)
          (setf (aref probs tt j)
                (exp (- (aref logits tt j) lse))))))
    (values (/ total (float t-len 0.0f0))
            (list :probs probs :target-ids target-ids))))

(defun softmax-ce-bwd (saved)
  "Backward for fused softmax+CE. Consumes SAVED from SOFTMAX-CE-FWD.
   Returns GRAD-LOGITS with same shape as PROBS:
     grad[t, k] = (1/T) * (probs[t, k] - I[k = target_t])

   Worked example (T=1 so the 1/T scale is just 1, V=3, target=1):
   if the model's softmax probs for this token were [0.2, 0.5, 0.3]
   (so it gave the correct class, index 1, only 50% probability), the
   gradient is probs minus a one-hot at the target index:
       grad = [0.2, 0.5, 0.3] - [0, 1, 0] = [0.2, -0.5, 0.3]
   Negative at the target (pushes that logit UP, since Adam descends
   the gradient), positive everywhere else (pushes those logits DOWN)
   — exactly the direction that increases probs[target] at the expense
   of the other classes. Implemented as: start every column at
   scale*probs[t,k] (loop below), then DECF just the target column by
   the same scale — cheaper than building an explicit one-hot tensor."
  (let* ((probs      (getf saved :probs))
         (target-ids (getf saved :target-ids))
         (t-len (array-dimension probs 0))
         (v     (array-dimension probs 1))
         (grad  (tensor-like probs))
         (scale (/ 1.0f0 (float t-len 0.0f0))))
    (declare (type tensor-2d probs grad)
             (type (simple-array fixnum (*)) target-ids)
             (type fixnum t-len v)
             (type single-float scale))
    (dotimes (tt t-len)
      (let ((tgt (aref target-ids tt)))
        (declare (type fixnum tgt))
        (dotimes (j v)
          (setf (aref grad tt j) (* scale (aref probs tt j))))
        (decf (aref grad tt tgt) scale)))
    grad))


;;; ------------------------------------------------------------------
;;; Nested-plist walkers for model parameters
;;; ------------------------------------------------------------------
;;;
;;; Model params are a plist. Leaves are SIMPLE-ARRAY SINGLE-FLOAT tensors.
;;; Non-leaf slots hold either nested plists (transformer-block-style) or
;;; scalars like :n-heads / :eps / :scale that must be preserved but never
;;; treated as trainable.
;;;
;;; Convention: we skip
;;;   - :PE (positional-sinusoidal's fixed table; not trainable by design)
;;;   - anything that isn't a single-float array
;;;   - anything whose value isn't a keyword-headed list (so we don't
;;;     accidentally recurse into random lists that happen to look like plists)
;;;
;;; COLLECT-TENSORS gives a deterministic flat list of trainable tensors;
;;; called on params, grads, m and v in parallel it yields matching order.
;;;
;;; PARAMS-MAP-TENSORS produces a new plist mirroring the original's
;;; structure, with FN applied to each trainable tensor. Used to allocate
;;; zero-filled buffers for Adam moments.

(defun collect-tensors (params)
  "Flat list of every trainable single-float tensor leaf in PARAMS,
   in the deterministic preorder in which they appear in the plist."
  (loop for (k v) on params by #'cddr
        append (cond
                 ((eq k :pe) nil)                             ; fixed, skip
                 ((and (arrayp v)
                       (eq (array-element-type v) 'single-float))
                  (list v))
                 ((and (listp v) (not (null v))
                       (keywordp (first v)))                  ; nested plist
                  (collect-tensors v))
                 (t nil))))                                   ; scalar, skip

(defun params-map-tensors (fn params)
  "Return a new plist mirroring PARAMS's shape, with FN applied to every
   trainable tensor leaf. Non-tensor entries (scalars, :PE) pass through
   unchanged so the returned structure can be walked in the same order."
  (loop for (k v) on params by #'cddr
        collect k
        collect (cond
                  ((eq k :pe) v)
                  ((and (arrayp v)
                        (eq (array-element-type v) 'single-float))
                   (funcall fn v))
                  ((and (listp v) (not (null v))
                        (keywordp (first v)))
                   (params-map-tensors fn v))
                  (t v))))


;;; ------------------------------------------------------------------
;;; Adam optimizer
;;; ------------------------------------------------------------------
;;;
;;; State layout:
;;;   (:m <mirrors params> :v <mirrors params> :step <fixnum>)
;;; where :m and :v are plists with the same shape as PARAMS but each
;;; trainable tensor replaced by a zero-filled buffer.
;;;
;;; Update rule (Kingma & Ba 2014, with bias correction):
;;;   m <- beta1 * m + (1 - beta1) * g
;;;   v <- beta2 * v + (1 - beta2) * g^2
;;;   m_hat <- m / (1 - beta1^t)
;;;   v_hat <- v / (1 - beta2^t)
;;;   param <- param - lr * m_hat / (sqrt(v_hat) + eps)

(defparameter *adam-beta1* 0.9f0)
(defparameter *adam-beta2* 0.999f0)
(defparameter *adam-eps*   1.0f-8)

(defun adam-alloc (params)
  "Allocate zeroed Adam state (moment buffers + step counter) mirroring
   PARAMS. Two separate walks so :m and :v get distinct buffers."
  (list :m    (params-map-tensors (lambda (tt) (zeros (array-dimensions tt))) params)
        :v    (params-map-tensors (lambda (tt) (zeros (array-dimensions tt))) params)
        :step 0))

(defun adam-step! (params grads state &key (lr 1.0f-3)
                                           (beta1 *adam-beta1*)
                                           (beta2 *adam-beta2*)
                                           (eps *adam-eps*))
  "Apply one Adam update. Mutates PARAMS and STATE in place. GRADS must
   mirror PARAMS's structure so that COLLECT-TENSORS returns matching order.

   Why bias-correct (B1T/B2T): M and V both start at zero and are
   exponential moving averages, so early on — before enough steps have
   accumulated — they're biased toward zero too (e.g. at STEP=1, M is
   only 10% of the true gradient, since M = 0.9*0 + 0.1*g). Dividing
   by (1 - beta^step) exactly cancels that bias (it's 1 at step 1 and
   approaches 1 as step grows), so the effective learning rate doesn't
   silently start too small and ramp up — without this correction,
   early training would move much slower than LR suggests."
  (declare (type single-float lr beta1 beta2 eps))
  (incf (getf state :step))
  (let* ((step  (getf state :step))
         (b1t   (- 1.0f0 (expt beta1 step)))
         (b2t   (- 1.0f0 (expt beta2 step)))
         (ps    (collect-tensors params))
         (gs    (collect-tensors grads))
         (ms    (collect-tensors (getf state :m)))
         (vs    (collect-tensors (getf state :v))))
    (declare (type fixnum step)
             (type single-float b1t b2t))
    (mapc
     (lambda (p g m v)
       (declare (type tensor p g m v))
       (dotimes (i (array-total-size p))
         (let* ((gi   (row-major-aref g i))
                (mi   (+ (* beta1 (row-major-aref m i))
                         (* (- 1.0f0 beta1) gi)))
                (vi   (+ (* beta2 (row-major-aref v i))
                         (* (- 1.0f0 beta2) gi gi)))
                (mhat (/ mi b1t))
                (vhat (/ vi b2t)))
           (declare (type single-float gi mi vi mhat vhat))
           (setf (row-major-aref m i) mi)
           (setf (row-major-aref v i) vi)
           (decf (row-major-aref p i)
                 (/ (* lr mhat) (+ (sqrt vhat) eps))))))
     ps gs ms vs)))


;;; ------------------------------------------------------------------
;;; Training-step composer
;;; ------------------------------------------------------------------
;;;
;;; The model itself is not defined yet (that's model.lisp, which will
;;; expand a DSL into a chain of module calls). To keep training.lisp
;;; independent, we take MODEL-FWD and MODEL-BWD as closures:
;;;
;;;   MODEL-FWD : (params, input) -> (values logits saved)
;;;   MODEL-BWD : (params, saved, grad-logits) -> (values _grad-input grads)
;;;
;;; INPUT is a (simple-array fixnum (T)) of token ids; TARGET-IDS is the
;;; same shape (usually input shifted by one). Grad w.r.t. input is
;;; irrelevant here (ids are discrete) — we discard it.

(defun train-step! (model-fwd model-bwd params state input target-ids
                    &key (lr 1.0f-3))
  "One complete training step — the full forward/loss/backward/update
   cycle used by every training loop in this project (TRAIN-LOOP in
   experiment.lisp calls this once per step). In order:
     1. MODEL-FWD:    input token ids  -> logits (T V), plus SAVED
                       state each layer's backward will need.
     2. SOFTMAX-CE-FWD: logits + target ids -> scalar LOSS, plus SAVED
                       softmax probabilities for the backward.
     3. SOFTMAX-CE-BWD: turns that into GRAD-LOGITS (T V) — the
                       gradient the rest of the network needs to start
                       backpropagating from.
     4. MODEL-BWD:    threads GRAD-LOGITS backward through every layer
                       (in reverse), producing a GRADS tree shaped like
                       PARAMS. The gradient w.r.t. the model's INPUT
                       (first return value) is discarded — token ids
                       are discrete, so there's nothing to do with it.
     5. ADAM-STEP!:   applies GRADS to PARAMS in place, mutating STATE's
                       moment buffers too.
   Mutates PARAMS and STATE. Returns the scalar LOSS from step 2, for
   the caller to log/average."
  (multiple-value-bind (logits saved) (funcall model-fwd params input)
    (multiple-value-bind (loss ce-saved) (softmax-ce-fwd logits target-ids)
      (let* ((grad-logits (softmax-ce-bwd ce-saved))
             (grads       (nth-value 1 (funcall model-bwd params saved grad-logits))))
        (adam-step! params grads state :lr lr)
        loss))))


;;; ------------------------------------------------------------------
;;; Tests
;;; ------------------------------------------------------------------
;;;
;;; TEST-SOFTMAX-CE: numerically checks softmax-ce-bwd against central
;;; differences of softmax-ce-fwd, using check-close from gradcheck.lisp.
;;;
;;; TEST-ADAM: fits x to minimize L = 0.5 * sum(x^2), where the analytic
;;; gradient is just x itself. After enough Adam steps, x should be tiny.
;;; This exercises both walkers (via nested plist) and the update rule.

(defun test-softmax-ce ()
  "Numerical check of softmax-CE backward. PASS means the fused loss
   backward agrees with central differences of the fused loss forward."
  (format t "~&Test: softmax-CE gradient check...~%")
  (let* ((t-len 5)
         (v     8)
         (logits (random-tensor (list t-len v) :scale 1.0f0))
         (target-ids (make-array t-len :element-type 'fixnum
                                       :initial-contents
                                       (loop repeat t-len collect (random v)))))
    (multiple-value-bind (loss saved) (softmax-ce-fwd logits target-ids)
      (declare (ignore loss))
      (let ((analytic (softmax-ce-bwd saved)))
        (flet ((loss-fn ()
                 (nth-value 0 (softmax-ce-fwd logits target-ids))))
          (let ((numerical (numerical-grad-tensor #'loss-fn logits)))
            (multiple-value-bind (pass-p max-abs max-rel)
                (check-close analytic numerical)
              (format t "  max-abs ~,2E  rel ~,2E  ~A~%"
                      max-abs max-rel (if pass-p "PASS" "FAIL"))
              pass-p)))))))

(defun test-adam ()
  "Sanity check: fit x to minimise L = 0.5 * sum(x^2). Analytic grad = x.
   Uses a nested plist to exercise the walker. After a few hundred Adam
   steps from a random init, ||x|| should be near zero."
  (format t "~&Test: Adam drives quadratic to zero...~%")
  (let* ((x      (random-tensor '(3 4) :scale 1.0f0))
         (params (list :block (list :w x)))                  ; nested plist
         (grads  (list :block (list :w (tensor-like x))))    ; will refill each step
         (state  (adam-alloc params))
         (grad-w (getf (getf grads :block) :w)))
    (let ((init-norm (loop for i below (array-total-size x)
                           sum (expt (row-major-aref x i) 2))))
      (dotimes (_ 500)
        ;; grad = x (analytic gradient of 0.5*sum(x^2))
        (dotimes (i (array-total-size x))
          (setf (row-major-aref grad-w i) (row-major-aref x i)))
        (adam-step! params grads state :lr 1.0f-2))
      (let ((final-norm (loop for i below (array-total-size x)
                              sum (expt (row-major-aref x i) 2))))
        (format t "  ||x||^2  init=~,4E  final=~,4E  ~A~%"
                init-norm final-norm
                (if (< final-norm 1.0f-4) "PASS" "FAIL"))
        (< final-norm 1.0f-4)))))

(defun run-training-tests ()
  "Run all training-module sanity checks. Returns T iff all PASS."
  (format t "~&=== Training module tests ===~%")
  (let ((r1 (test-softmax-ce))
        (r2 (test-adam)))
    (format t "~&=== Done ===~%")
    (and r1 r2)))
