;;;; src/tensor-ops.lisp
;;;;
;;;; Primitive tensor operations. Every op is a paired (forward, backward)
;;;; function. FORWARD returns (values output saved) where SAVED is
;;;; whatever the backward needs from the forward pass (usually the inputs,
;;;; sometimes intermediate values). BACKWARD takes (grad-out saved) and
;;;; returns gradients w.r.t. each input.
;;;;
;;;; Convention for shapes:
;;;;   - Vectors: (simple-array single-float (N))
;;;;   - Matrices: (simple-array single-float (M N)), row-major
;;;;   - MATMUL uses y = W . x convention: W is (M K), x is (K N), y is (M N)
;;;;     When x is a single vector of length K, treat as (K 1) or use MATVEC.
;;;;
;;;; Backward output convention:
;;;;   - Ops return NEW gradient tensors (not accumulated into existing ones).
;;;;     The training loop / optimizer is responsible for accumulation.
;;;;   - This keeps ops pure and easy to test; the cost is one allocation
;;;;     per op per step, which is fine at this model scale.
;;;;
;;;; Activation choice: GELU (Gaussian Error Linear Unit).
;;;;   Forward:  gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
;;;;   We use the tanh approximation from the original GELU paper:
;;;;     gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
;;;;   because CL doesn't ship ERF and tanh is standard. This is the same
;;;;   approximation used in the original BERT/GPT-2 codebases.
;;;;
;;;; How to run (from project root):
;;;;   (load (compile-file "src/utilities.lisp"))
;;;;   (load (compile-file "src/tensor-ops.lisp"))

;;; ---------- Type declarations for speed ----------
;;; SBCL benefits substantially from knowing that tensors are simple-arrays
;;; of single-float. We declare the type once here so ops can use it.

(deftype tensor () '(simple-array single-float))
(deftype tensor-1d () '(simple-array single-float (*)))
(deftype tensor-2d () '(simple-array single-float (* *)))


;;; ==========================================================================
;;; MATMUL : y = A . B   where A is (M K), B is (K N), y is (M N)
;;; ==========================================================================
;;;
;;; Forward: standard triple loop. No blocking / SIMD — that's an optimization
;;; for later if we need it. At d_model=64 with modest context, plain loops
;;; on SBCL with type declarations are fast enough.
;;;
;;; Backward:
;;;   grad_A = grad_out . B^T     shape (M K)
;;;   grad_B = A^T . grad_out     shape (K N)
;;; Derivation: for y[i,j] = sum_k A[i,k] * B[k,j],
;;;   dy[i,j]/dA[p,q] = B[q,j] if i=p else 0
;;;   so d L / d A[p,q] = sum_j (grad_out[p,j] * B[q,j]) = (grad_out . B^T)[p,q]

(defun matmul-forward (a b)
  "y = A . B. A is (M K), B is (K N), y is (M N).
   Saves (A B) for backward."
  (declare (type tensor-2d a b))
  (let* ((m (array-dimension a 0))
         (k (array-dimension a 1))
         (n (array-dimension b 1))
         (y (make-tensor (list m n))))
    (declare (type fixnum m k n)
             (type tensor-2d y))
    (assert (= k (array-dimension b 0)) ()
            "MATMUL shape mismatch: A is ~A, B is ~A"
            (array-dimensions a) (array-dimensions b))
    (dotimes (i m)
      (dotimes (j n)
        (let ((acc 0.0f0))
          (declare (type single-float acc))
          (dotimes (kk k)
            (incf acc (* (aref a i kk) (aref b kk j))))
          (setf (aref y i j) acc))))
    (values y (list a b))))

(defun matmul-backward (grad-out saved)
  "Returns (values grad-a grad-b)."
  (declare (type tensor-2d grad-out))
  (destructuring-bind (a b) saved
    (declare (type tensor-2d a b))
    (let* ((m (array-dimension a 0))
           (k (array-dimension a 1))
           (n (array-dimension b 1))
           (grad-a (make-tensor (list m k)))
           (grad-b (make-tensor (list k n))))
      (declare (type fixnum m k n)
               (type tensor-2d grad-a grad-b))
      ;; grad_A[i,q] = sum_j grad_out[i,j] * B[q,j]
      (dotimes (i m)
        (dotimes (q k)
          (let ((acc 0.0f0))
            (declare (type single-float acc))
            (dotimes (j n)
              (incf acc (* (aref grad-out i j) (aref b q j))))
            (setf (aref grad-a i q) acc))))
      ;; grad_B[q,j] = sum_i A[i,q] * grad_out[i,j]
      (dotimes (q k)
        (dotimes (j n)
          (let ((acc 0.0f0))
            (declare (type single-float acc))
            (dotimes (i m)
              (incf acc (* (aref a i q) (aref grad-out i j))))
            (setf (aref grad-b q j) acc))))
      (values grad-a grad-b))))


;;; ==========================================================================
;;; ADD : y = a + b   elementwise, same shape
;;; ==========================================================================
;;;
;;; No broadcasting in the base ADD — modules that need bias broadcasting
;;; use ADD-BIAS below. Keeps this op simple and its backward trivial.
;;;
;;; Backward: grad_a = grad_out, grad_b = grad_out. Both inputs pass the
;;; upstream gradient through unchanged.

(defun add-forward (a b)
  "Elementwise a + b. Shapes must match."
  (declare (type tensor a b))
  (assert (equal (array-dimensions a) (array-dimensions b)))
  (let ((y (tensor-like a)))
    (declare (type tensor y))
    (dotimes (i (array-total-size a))
      (setf (row-major-aref y i)
            (+ (row-major-aref a i) (row-major-aref b i))))
    ;; Nothing to save — the backward doesn't depend on inputs.
    (values y nil)))

(defun add-backward (grad-out saved)
  "Returns (values grad-a grad-b). Both equal grad-out (fresh copies)."
  (declare (ignore saved))
  (declare (type tensor grad-out))
  (let ((ga (tensor-like grad-out))
        (gb (tensor-like grad-out)))
    (dotimes (i (array-total-size grad-out))
      (setf (row-major-aref ga i) (row-major-aref grad-out i))
      (setf (row-major-aref gb i) (row-major-aref grad-out i)))
    (values ga gb)))


;;; ==========================================================================
;;; ADD-BIAS : y[i,j] = x[i,j] + b[j]   for x (M N), b (N)
;;; ==========================================================================
;;;
;;; Backward:
;;;   grad_x = grad_out (shape preserved)
;;;   grad_b[j] = sum_i grad_out[i,j]   (gradient collapses along broadcast axis)

(defun add-bias-forward (x b)
  "y[i,j] = x[i,j] + b[j]. x is (M N), b is (N)."
  (declare (type tensor-2d x)
           (type tensor-1d b))
  (let ((m (array-dimension x 0))
        (n (array-dimension x 1))
        (y (tensor-like x)))
    (declare (type fixnum m n)
             (type tensor-2d y))
    (assert (= n (length b)))
    (dotimes (i m)
      (dotimes (j n)
        (setf (aref y i j) (+ (aref x i j) (aref b j)))))
    (values y nil)))

(defun add-bias-backward (grad-out saved)
  "Returns (values grad-x grad-b)."
  (declare (ignore saved))
  (declare (type tensor-2d grad-out))
  (let* ((m (array-dimension grad-out 0))
         (n (array-dimension grad-out 1))
         (grad-x (tensor-like grad-out))
         (grad-b (make-tensor (list n))))
    (declare (type fixnum m n)
             (type tensor-2d grad-x)
             (type tensor-1d grad-b))
    ;; grad_x = grad_out (elementwise copy)
    (dotimes (i (* m n))
      (setf (row-major-aref grad-x i) (row-major-aref grad-out i)))
    ;; grad_b[j] = sum_i grad_out[i,j]
    (dotimes (j n)
      (let ((acc 0.0f0))
        (declare (type single-float acc))
        (dotimes (i m)
          (incf acc (aref grad-out i j)))
        (setf (aref grad-b j) acc)))
    (values grad-x grad-b)))


;;; ==========================================================================
;;; SCALE : y = alpha * x   scalar alpha, tensor x
;;; ==========================================================================
;;;
;;; ALPHA is treated as a constant (not a learned parameter). Only x gets a
;;; gradient. Used e.g. for the 1/sqrt(d_k) factor in attention.
;;;
;;; Backward: grad_x = alpha * grad_out.

(defun scale-forward (x alpha)
  "y = alpha * x elementwise."
  (declare (type tensor x)
           (type single-float alpha))
  (let ((y (tensor-like x)))
    (dotimes (i (array-total-size x))
      (setf (row-major-aref y i)
            (* alpha (row-major-aref x i))))
    (values y alpha)))

(defun scale-backward (grad-out saved)
  "Returns grad_x = alpha * grad_out."
  (declare (type tensor grad-out)
           (type single-float saved))
  (let ((gx (tensor-like grad-out)))
    (dotimes (i (array-total-size grad-out))
      (setf (row-major-aref gx i)
            (* saved (row-major-aref grad-out i))))
    gx))


;;; ==========================================================================
;;; SOFTMAX : along the last axis of a 2D tensor
;;; ==========================================================================
;;;
;;; For x of shape (M N), softmax is computed independently along each row.
;;; Numerical stability: subtract row max before exp.
;;;
;;;   y[i,j] = exp(x[i,j] - max_k x[i,k]) / sum_k exp(x[i,k] - max_l x[i,l])
;;;
;;; Backward: the softmax Jacobian for row i is
;;;   dy[i,j]/dx[i,k] = y[i,j] * (delta_jk - y[i,k])
;;; Multiplied against grad_out:
;;;   grad_x[i,k] = y[i,k] * (grad_out[i,k] - sum_j grad_out[i,j] * y[i,j])
;;; This is the standard "softmax backward" formula. Row-independent.
;;;
;;; We save Y (the softmax output) — the backward doesn't need X.

(defun softmax-forward (x)
  "Row-wise softmax on 2D X of shape (M N)."
  (declare (type tensor-2d x))
  (let* ((m (array-dimension x 0))
         (n (array-dimension x 1))
         (y (tensor-like x)))
    (declare (type fixnum m n)
             (type tensor-2d y))
    (dotimes (i m)
      ;; Row max for numerical stability.
      (let ((row-max most-negative-single-float))
        (declare (type single-float row-max))
        (dotimes (j n)
          (when (> (aref x i j) row-max)
            (setf row-max (aref x i j))))
        ;; Compute exp(x - max) and running sum.
        (let ((sum 0.0f0))
          (declare (type single-float sum))
          (dotimes (j n)
            (let ((e (exp (- (aref x i j) row-max))))
              (declare (type single-float e))
              (setf (aref y i j) e)
              (incf sum e)))
          ;; Normalize.
          (let ((inv-sum (/ 1.0f0 sum)))
            (declare (type single-float inv-sum))
            (dotimes (j n)
              (setf (aref y i j) (* (aref y i j) inv-sum)))))))
    (values y y)))

(defun softmax-backward (grad-out saved)
  "Returns grad_x. SAVED is the softmax output Y from forward."
  (declare (type tensor-2d grad-out)
           (type tensor-2d saved))
  (let* ((m (array-dimension grad-out 0))
         (n (array-dimension grad-out 1))
         (grad-x (tensor-like grad-out)))
    (declare (type fixnum m n)
             (type tensor-2d grad-x))
    (dotimes (i m)
      ;; dot = sum_j grad_out[i,j] * y[i,j]
      (let ((dot 0.0f0))
        (declare (type single-float dot))
        (dotimes (j n)
          (incf dot (* (aref grad-out i j) (aref saved i j))))
        ;; grad_x[i,k] = y[i,k] * (grad_out[i,k] - dot)
        (dotimes (k n)
          (setf (aref grad-x i k)
                (* (aref saved i k)
                   (- (aref grad-out i k) dot))))))
    grad-x))


;;; ==========================================================================
;;; GELU : y = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
;;; ==========================================================================
;;;
;;; Elementwise. The tanh approximation avoids needing ERF.
;;;
;;; Backward derivation (let c = sqrt(2/pi), a = 0.044715, u = c(x + a x^3)):
;;;   y = 0.5 x (1 + tanh u)
;;;   du/dx = c (1 + 3 a x^2)
;;;   dy/dx = 0.5 (1 + tanh u) + 0.5 x * sech^2(u) * du/dx
;;;         = 0.5 (1 + tanh u) + 0.5 x (1 - tanh^2 u) * c (1 + 3 a x^2)
;;;
;;; We save X (need it in the backward).

(defconstant +gelu-c+ (coerce (sqrt (/ 2.0d0 (float pi 1.0d0))) 'single-float))
(defconstant +gelu-a+ 0.044715f0)

(defun gelu-forward (x)
  "Elementwise GELU (tanh approximation)."
  (declare (type tensor x))
  (let ((y (tensor-like x)))
    (dotimes (i (array-total-size x))
      (let* ((xi (row-major-aref x i))
             (u  (* +gelu-c+ (+ xi (* +gelu-a+ xi xi xi))))
             (t- (tanh u)))
        (declare (type single-float xi u t-))
        (setf (row-major-aref y i)
              (* 0.5f0 xi (+ 1.0f0 t-)))))
    (values y x)))

(defun gelu-backward (grad-out saved)
  "Returns grad_x. SAVED is the original input X."
  (declare (type tensor grad-out)
           (type tensor saved))
  (let ((gx (tensor-like grad-out)))
    (dotimes (i (array-total-size grad-out))
      (let* ((xi (row-major-aref saved i))
             (u  (* +gelu-c+ (+ xi (* +gelu-a+ xi xi xi))))
             (t- (tanh u))
             (sech2 (- 1.0f0 (* t- t-)))
             (du/dx (* +gelu-c+ (+ 1.0f0 (* 3.0f0 +gelu-a+ xi xi))))
             (dy/dx (+ (* 0.5f0 (+ 1.0f0 t-))
                       (* 0.5f0 xi sech2 du/dx))))
        (declare (type single-float xi u t- sech2 du/dx dy/dx))
        (setf (row-major-aref gx i)
              (* dy/dx (row-major-aref grad-out i)))))
    gx))


;;; ==========================================================================
;;; TRANSPOSE : 2D only
;;; ==========================================================================
;;;
;;; Backward of transpose is transpose. No saved state needed beyond shape.

(defun transpose-forward (x)
  "Transpose 2D tensor X."
  (declare (type tensor-2d x))
  (let* ((m (array-dimension x 0))
         (n (array-dimension x 1))
         (y (make-tensor (list n m))))
    (declare (type fixnum m n)
             (type tensor-2d y))
    (dotimes (i m)
      (dotimes (j n)
        (setf (aref y j i) (aref x i j))))
    (values y nil)))

(defun transpose-backward (grad-out saved)
  "Returns grad_x = transpose(grad_out)."
  (declare (ignore saved))
  (declare (type tensor-2d grad-out))
  (multiple-value-bind (gx _) (transpose-forward grad-out)
    (declare (ignore _))
    gx))


;;; ==========================================================================
;;; EMBEDDING-LOOKUP : ids -> rows of embedding table
;;; ==========================================================================
;;;
;;; Table is (vocab-size, d_model). IDs is a length-T fixnum vector.
;;; Output is (T, d_model): row t of the output is table[ids[t], :].
;;;
;;; Backward: accumulates into a grad-table of shape (vocab-size, d_model).
;;;   For each t, grad-table[ids[t], :] += grad_out[t, :]
;;; IDs are not differentiable (they're indices), so only the table gets
;;; a gradient. Note: same id appearing at multiple positions accumulates.

(defun embedding-lookup-forward (table ids)
  "Look up rows of TABLE at IDS. Returns (T, d_model).
   Saves (table-shape ids) for backward — we don't save the table itself
   because the backward only needs its shape."
  (declare (type tensor-2d table)
           (type (simple-array fixnum (*)) ids))
  (let* ((vocab (array-dimension table 0))
         (d     (array-dimension table 1))
         (t-len (length ids))
         (y     (make-tensor (list t-len d))))
    (declare (type fixnum vocab d t-len)
             (type tensor-2d y))
    (dotimes (tt t-len)
      (let ((id (aref ids tt)))
        (declare (type fixnum id))
        (assert (< id vocab) () "Embedding id ~A out of vocab size ~A" id vocab)
        (dotimes (k d)
          (setf (aref y tt k) (aref table id k)))))
    (values y (list (array-dimensions table) ids))))

(defun embedding-lookup-backward (grad-out saved)
  "Returns grad_table of the original table shape. IDs get no gradient."
  (declare (type tensor-2d grad-out))
  (destructuring-bind (table-shape ids) saved
    (declare (type (simple-array fixnum (*)) ids))
    (let* ((d     (second table-shape))
           (t-len (length ids))
           (grad-table (make-tensor table-shape)))
      (declare (type fixnum d t-len)
               (type tensor-2d grad-table))
      (dotimes (tt t-len)
        (let ((id (aref ids tt)))
          (declare (type fixnum id))
          (dotimes (k d)
            (incf (aref grad-table id k) (aref grad-out tt k)))))
      grad-table)))
