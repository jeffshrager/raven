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

;; Combined absolute + relative tolerance (torch.autograd.gradcheck style).
;; Pass criterion, per element: |a - n| <= *gc-atol* + *gc-rtol* * max(|a|, |n|).
;; Rationale:
;;   The atol term covers the fixed absolute noise of single-float central
;;   differences, roughly L*epsilon/h. In our test setups this is around
;;   1e-4 to 2e-4 for the naive elementwise-loop forward pass; 3e-4 leaves
;;   safe margin for the longer chains in attention (Q/K flow through
;;   softmax, which is precision-hungry).
;;   The rtol term covers accumulated relative noise on larger-magnitude
;;   gradients. 5% is well below any real bug (a sign flip is 200%, an
;;   off-by-one is order 100%, a missing chain-rule term is order 100%).

(defparameter *gc-atol* 3.0f-4
  "Absolute tolerance for numerical vs analytic gradient comparison.")

(defparameter *gc-rtol* 5.0f-2
  "Relative tolerance (5%) applied on top of *GC-ATOL*.")

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

(defun check-close (analytic numerical)
  "Compare two gradient tensors elementwise using combined atol + rtol.
   Returns (values PASS-P MAX-ABS-DIFF MAX-REL-ERR).
     PASS-P      : T iff every element satisfies
                     |a - n| <= *GC-ATOL* + *GC-RTOL* * max(|a|, |n|).
     MAX-ABS-DIFF: max |a - n| over all elements. Direct measure of
                   agreement, meaningful independent of magnitude.
     MAX-REL-ERR : max of |a - n| / max(|a|, |n|) over elements where at
                   least one of |a|, |n| exceeds *GC-ATOL*. Purely
                   diagnostic; may look high on small-gradient elements
                   because central diff is noisy there, even though PASS-P
                   correctly says the element is fine."
  (declare (type tensor analytic numerical))
  (let ((all-pass t)
        (max-abs 0.0f0)
        (max-rel 0.0f0)
        (atol *gc-atol*)
        (rtol *gc-rtol*))
    (declare (type single-float max-abs max-rel atol rtol))
    (dotimes (i (array-total-size analytic))
      (let* ((a (row-major-aref analytic i))
             (n (row-major-aref numerical i))
             (aa (abs a)) (nn (abs n))
             (absdiff (abs (- a n))))
        (declare (type single-float a n aa nn absdiff))
        (when (> absdiff max-abs) (setf max-abs absdiff))
        (when (> absdiff (+ atol (* rtol (max aa nn))))
          (setf all-pass nil))
        (when (or (>= aa atol) (>= nn atol))
          (let ((rel (/ absdiff (max aa nn))))
            (declare (type single-float rel))
            (when (> rel max-rel) (setf max-rel rel))))))
    (values all-pass max-abs max-rel)))


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

            (let ((param-results nil))
              ;; Check each float-tensor param.
              ;; Each entry: (KEY PASS-P MAX-ABS-DIFF MAX-REL-ERR).
              (loop for (k v) on params by #'cddr do
                    (when (and (arrayp v)
                               (eq (array-element-type v) 'single-float)
                               (not (eq k :pe)))
                      (let* ((num-grad (numerical-grad-tensor #'loss v))
                             (ana-grad (getf analytic-grad-params k)))
                        (multiple-value-bind (pass-p max-abs max-rel)
                            (check-close ana-grad num-grad)
                          (push (list k pass-p max-abs max-rel) param-results)))))

              ;; Check grad w.r.t. input (only for float inputs).
              (let ((input-pass nil) (input-abs nil) (input-rel nil))
                (when (and (arrayp input)
                           (eq (array-element-type input) 'single-float)
                           analytic-grad-input)
                  (let ((num-grad (numerical-grad-tensor #'loss input)))
                    (multiple-value-bind (pass-p max-abs max-rel)
                        (check-close analytic-grad-input num-grad)
                      (setf input-pass pass-p input-abs max-abs input-rel max-rel))))

                (list :type type
                      :param-results (nreverse param-results)
                      :input-pass  input-pass
                      :input-abs   input-abs
                      :input-rel   input-rel)))))))))


;;; ---------- Report formatting ----------

(defun report-result (result)
  "Print one result line per parameter and per input.
   PASS/FAIL comes from CHECK-CLOSE's atol+rtol verdict.
   MAX-ABS and REL are diagnostic only; REL may look large on very small
   gradients but the PASS verdict correctly accounts for that via atol."
  (let ((type       (getf result :type))
        (presults   (getf result :param-results))
        (input-pass (getf result :input-pass))
        (input-abs  (getf result :input-abs))
        (input-rel  (getf result :input-rel)))
    (format t "~&~A~%" type)
    (dolist (pr presults)
      (destructuring-bind (k pass-p max-abs max-rel) pr
        (format t "  param ~A : max-abs ~,2E  rel ~,2E  ~A~%"
                k max-abs max-rel (if pass-p "PASS" "FAIL"))))
    (when input-abs
      (format t "  input   : max-abs ~,2E  rel ~,2E  ~A~%"
              input-abs input-rel (if input-pass "PASS" "FAIL")))))


;;; ---------- Suite ----------

(defun run-all-gradchecks ()
  "Run gradient check on every registered module (except composite ones
   whose params are nested plists — those get end-to-end checks later)."
  (format t "~&=== Gradient checks (h=~,1E atol=~,1E rtol=~,1E) ===~%"
          *gc-h* *gc-atol* *gc-rtol*)

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
