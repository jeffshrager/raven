;;;; src/gradcheck.lisp
;;;;
;;;; Per-module numerical gradient verification. For each module type, we:
;;;;   1. Build a small random input and random params (via the module's
;;;;      allocator, then jittered so gradients aren't at trivial values).
;;;;   2. Run forward, record output.
;;;;   3. Pick a random upstream gradient (same shape as output).
;;;;   4. Compute analytic gradients via the module's backward.
;;;;   5. Compute numerical gradients by central differences on a scalar
;;;;      loss L = sum(output * upstream). Then dL/dp = analytic gradient
;;;;      (by construction, since L is linear in output).
;;;;   6. Compare analytic vs numerical, report max relative error.
;;;;
;;;; Why L = sum(output * upstream):
;;;;   dL/dy[i]     = upstream[i]                (by definition of L)
;;;;   dL/dparam    = (dL/dy) . (dy/dparam)
;;;;                = upstream . (dy/dparam)     which is exactly what
;;;;   backward computes when given upstream as grad_out. So comparing
;;;;   backward's output against numerical dL/dparam is a valid check.
;;;;
;;;; Precision expectations (single-float, central differences, h=1e-3):
;;;;   - Relative error < 1e-2 : PASS
;;;;   - Relative error 1e-2 to 1e-1 : suspicious, investigate
;;;;   - Relative error > 1e-1 : FAIL, likely bug
;;;;
;;;; How to run (from project root):
;;;;   (load (compile-file "src/utilities.lisp"))
;;;;   (load (compile-file "src/tensor-ops.lisp"))
;;;;   (load (compile-file "src/modules.lisp"))
;;;;   (load (compile-file "src/gradcheck.lisp"))
;;;;   (run-all-gradchecks)


;;; ---------- Config ----------

(defparameter *gc-h* 1.0f-3
  "Perturbation size for finite differences.")

(defparameter *gc-tol* 1.0f-2
  "Relative-error tolerance for PASS. Loose because of single-float.")

(defparameter *gc-seed-random-state* nil
  "If set to a random-state, we reset *random-state* to a copy of it
   before each check. Useful for reproducibility. NIL = don't reset.")


;;; ---------- Small helpers ----------

(defun fill-random! (tensor &key (scale 1.0f0))
  "Fill TENSOR with random values in [-scale, +scale]."
  (declare (type single-float scale))
  (dotimes (i (array-total-size tensor))
    (setf (row-major-aref tensor i)
          (* scale (- (random 2.0f0) 1.0f0))))
  tensor)

(defun random-tensor (dims &key (scale 1.0f0))
  (fill-random! (make-tensor dims) :scale scale))

(defun dot-tensors (a b)
  "Sum of elementwise product. A and B must have same total size."
  (declare (type tensor a b))
  (let ((acc 0.0f0))
    (declare (type single-float acc))
    (dotimes (i (array-total-size a))
      (incf acc (* (row-major-aref a i) (row-major-aref b i))))
    acc))

(defun max-rel-error (analytic numerical)
  "Max over elements of |a - n| / max(|a|, |n|, epsilon).
   ANALYTIC and NUMERICAL must have same shape."
  (declare (type tensor analytic numerical))
  (let ((max-err 0.0f0)
        (eps 1.0f-6))
    (declare (type single-float max-err eps))
    (dotimes (i (array-total-size analytic))
      (let* ((a (row-major-aref analytic i))
             (n (row-major-aref numerical i))
             (denom (max (abs a) (abs n) eps))
             (err (/ (abs (- a n)) denom)))
        (declare (type single-float a n denom err))
        (when (> err max-err) (setf max-err err))))
    max-err))


;;; ---------- Core numerical check for one parameter tensor ----------

(defun numerical-grad-tensor (loss-fn param)
  "Compute numerical gradient of LOSS-FN with respect to PARAM using
   central differences. LOSS-FN takes no args and returns a scalar
   single-float; it must read PARAM's current values.
   PARAM is mutated (perturbed then restored) during the check.
   Returns a fresh tensor of PARAM's shape."
  (declare (type tensor param))
  (let ((grad (tensor-like param))
        (h *gc-h*))
    (declare (type single-float h)
             (type tensor grad))
    (dotimes (i (array-total-size param))
      (let ((orig (row-major-aref param i)))
        (declare (type single-float orig))
        (setf (row-major-aref param i) (+ orig h))
        (let ((l+ (funcall loss-fn)))
          (setf (row-major-aref param i) (- orig h))
          (let ((l- (funcall loss-fn)))
            (setf (row-major-aref param i) orig)
            (setf (row-major-aref grad i)
                  (/ (- l+ l-) (* 2.0f0 h)))))))
    grad))


;;; ---------- Generic per-module check ----------

(defun check-module (type config input-dims
                     &key (input-scale 0.5f0)
                          (jitter-params-scale 0.1f0))
  "Run gradient check on module TYPE with CONFIG. INPUT-DIMS is either a
   list of ints (float input) or the keyword :IDS (integer input for
   :embedding). INPUT-SCALE controls random-input magnitude.
   JITTER-PARAMS-SCALE adds noise to allocated params so they aren't at
   symmetric zero (which can mask bugs).

   Returns a plist (:type ... :param-errors ((:key error) ...) :input-error ...)."

  ;; Reset RNG if requested (reproducibility).
  (when *gc-seed-random-state*
    (setf *random-state* (make-random-state *gc-seed-random-state*)))

  (let* ((params (module-alloc type config))
         (input  (if (eq input-dims :ids)
                     ;; Integer ids for embedding module.
                     (let* ((vocab (getf config :vocab-size))
                            (t-len 5)
                            (arr (make-array t-len :element-type 'fixnum)))
                       (dotimes (i t-len)
                         (setf (aref arr i) (random vocab)))
                       arr)
                     (random-tensor input-dims :scale input-scale))))

    ;; Jitter float params (skip non-tensor entries like :n-heads, :eps, :pe).
    (loop for (k v) on params by #'cddr do
          (when (and (arrayp v)
                     (eq (array-element-type v) 'single-float)
                     ;; Don't jitter the sinusoidal PE (it's a fixed constant).
                     (not (eq k :pe)))
            (dotimes (i (array-total-size v))
              (incf (row-major-aref v i)
                    (* jitter-params-scale (- (random 2.0f0) 1.0f0))))))

    ;; Nested-params modules (transformer-block) need recursive jitter — skip
    ;; here for simplicity; those get exercised by end-to-end tests later.

    (multiple-value-bind (output saved) (module-forward type params input nil)
      (let* ((upstream (random-tensor (array-dimensions output) :scale 1.0f0)))

        ;; Analytic gradients.
        (multiple-value-bind (analytic-grad-input analytic-grad-params)
            (module-backward type params saved upstream)

          ;; Loss closure: L = sum(output(param) * upstream)
          ;; We rebuild forward each call — modules are pure functions of
          ;; their params + input, so re-running forward with mutated params
          ;; gives the correct new output.
          (flet ((loss ()
                   (multiple-value-bind (out _saved) (module-forward type params input nil)
                     (declare (ignore _saved))
                     (dot-tensors out upstream))))

            (let ((param-errors nil))
              ;; Check each float-tensor param.
              (loop for (k v) on params by #'cddr do
                    (when (and (arrayp v)
                               (eq (array-element-type v) 'single-float)
                               (not (eq k :pe)))
                      (let* ((num-grad (numerical-grad-tensor #'loss v))
                             (ana-grad (getf analytic-grad-params k))
                             (err (max-rel-error ana-grad num-grad)))
                        (push (list k err) param-errors))))

              ;; Check grad w.r.t. input (only for float inputs).
              (let ((input-error nil))
                (when (and (arrayp input)
                           (eq (array-element-type input) 'single-float)
                           analytic-grad-input)
                  (let ((num-grad (numerical-grad-tensor #'loss input)))
                    (setf input-error (max-rel-error analytic-grad-input num-grad))))

                (list :type type
                      :param-errors (nreverse param-errors)
                      :input-error input-error)))))))))


;;; ---------- Report formatting ----------

(defun report-result (result)
  "Print one result line per parameter and per input."
  (let ((type   (getf result :type))
        (perrs  (getf result :param-errors))
        (inperr (getf result :input-error))
        (tol    *gc-tol*))
    (format t "~&~A~%" type)
    (dolist (pe perrs)
      (destructuring-bind (k err) pe
        (format t "  param ~A : rel-err ~,4E  ~A~%"
                k err (if (< err tol) "PASS" "FAIL"))))
    (when inperr
      (format t "  input   : rel-err ~,4E  ~A~%"
              inperr (if (< inperr tol) "PASS" "FAIL")))))


;;; ---------- Suite ----------

(defun run-all-gradchecks ()
  "Run gradient check on every registered module (except composite ones
   whose params are nested plists — those get end-to-end checks later)."
  (format t "~&=== Gradient checks (h=~,1E tol=~,1E) ===~%" *gc-h* *gc-tol*)

  (dolist (case
           '(;; (module-type config input-dims-or-:ids)
             (:embedding
              (:vocab-size 20 :d-model 8)
              :ids)
             (:positional-sinusoidal
              (:d-model 8 :max-len 16)
              (5 8))
             (:rmsnorm
              (:d-model 8 :eps 1.0f-5)
              (5 8))
             (:attention-block
              (:d-model 8 :n-heads 2)
              (4 8))
             (:ffn
              (:d-model 8 :d-ff 16)
              (4 8))
             (:unembedding
              (:d-model 8 :vocab-size 20)
              (4 8))))
    (destructuring-bind (type config input-dims) case
      (report-result (check-module type config input-dims))))

  (format t "~&=== Done ===~%")
  (format t "Note: :transformer-block has nested params and is not covered here.~%")
  (format t "      Cover it with an end-to-end check once training scaffold exists.~%"))


;;; ---------- Entry point when loaded as a script ----------
;;; Uncomment to auto-run on load:
;;; (run-all-gradchecks)
