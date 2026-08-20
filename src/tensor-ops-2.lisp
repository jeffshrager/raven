;;;; tensor-ops-2.lisp
;;;; Second batch of tensor primitives: batched matmul, RMSNorm, GELU.
;;;;
;;;; How to run:
;;;;   (load (compile-file "src/tensor-ops.lisp"))       ; load first batch
;;;;   (load (compile-file "src/tensor-ops-2.lisp"))     ; then this one
;;;;   Tests for both batches run at load time.
;;;;
;;;; Assumes tensor-ops.lisp is already loaded (uses tensor struct,
;;;; make-zeros, make-from-list, tensor-ref, approx=).

;;; ------------------------------------------------------------------
;;; Batched matmul (3D x 3D -> 3D)
;;; ------------------------------------------------------------------
;;; A: (b m k), B: (b k n) -> C: (b m n).
;;; Leading dim is the batch (used for the "heads" axis in multi-head
;;; attention). Each of the b slices is an independent 2D matmul.
;;;
;;; Implementation: outer batch loop wraps the same i-p-j loop from matmul,
;;; with per-batch base offsets computed once. Not a call to matmul because
;;; the offset arithmetic is cleaner as one unified index expression.

(defun bmatmul (a b)
  "Batched matmul. A: (b m k), B: (b k n) -> result: (b m n)."
  (let ((ashape (tensor-shape a))
        (bshape (tensor-shape b)))
    (assert (and (= 3 (length ashape)) (= 3 (length bshape))) ()
            "bmatmul: both args must be 3D, got shapes ~A and ~A." ashape bshape)
    (let ((ba (first  ashape))
          (m  (second ashape))
          (k  (third  ashape))
          (bb (first  bshape))
          (kb (second bshape))
          (n  (third  bshape)))
      (assert (= ba bb) ()
              "bmatmul: batch dims disagree: ~A vs ~A." ashape bshape)
      (assert (= k kb) ()
              "bmatmul: inner dims disagree: ~A vs ~A." ashape bshape)
      (let* ((out (make-zeros (list ba m n)))
             (ad  (tensor-data a))
             (bd  (tensor-data b))
             (od  (tensor-data out))
             (a-stride (the fixnum (* m k)))    ; elements per A slice
             (b-stride (the fixnum (* k n)))    ; elements per B slice
             (o-stride (the fixnum (* m n))))   ; elements per output slice
        (declare (type (simple-array single-float (*)) ad bd od)
                 (type fixnum ba m k n a-stride b-stride o-stride)
                 (optimize (speed 3) (safety 1)))
        (dotimes (bi ba)
          (let ((a-base (the fixnum (* bi a-stride)))
                (b-base (the fixnum (* bi b-stride)))
                (o-base (the fixnum (* bi o-stride))))
            (dotimes (i m)
              (let ((ai-k (the fixnum (+ a-base (the fixnum (* i k)))))
                    (oi-n (the fixnum (+ o-base (the fixnum (* i n))))))
                (dotimes (p k)
                  (let ((aip  (aref ad (the fixnum (+ ai-k p))))
                        (bp-n (the fixnum (+ b-base (the fixnum (* p n))))))
                    (dotimes (j n)
                      (incf (aref od (the fixnum (+ oi-n j)))
                            (* aip (aref bd (the fixnum (+ bp-n j))))))))))))
        out))))

;;; ------------------------------------------------------------------
;;; RMSNorm (forward, in-place)
;;; ------------------------------------------------------------------
;;; For each row x of 2D input (m n):
;;;   rms = sqrt(mean(x_i^2) + eps)
;;;   y_i = (x_i / rms) * gain_i
;;; GAIN is a learned parameter of shape (n).
;;; No mean subtraction, no bias — that's what makes it simpler than LayerNorm.
;;;
;;; Epsilon default 1e-6: enough to avoid divide-by-zero without measurably
;;; shifting normal values. Standard choice for single-float.

(defun rmsnorm! (a gain &key (eps 1.0e-6))
  "In-place row-wise RMSNorm. A: (m n), GAIN: (n)."
  (let ((ashape (tensor-shape a))
        (gshape (tensor-shape gain)))
    (assert (and (= 2 (length ashape)) (= 1 (length gshape))) ()
            "rmsnorm!: need 2D input and 1D gain, got ~A and ~A." ashape gshape)
    (let ((m  (first  ashape))
          (n  (second ashape))
          (ng (first  gshape)))
      (assert (= n ng) ()
              "rmsnorm!: gain length ~D must match input cols ~D." ng n)
      (let ((ad     (tensor-data a))
            (gd     (tensor-data gain))
            (eps-sf (coerce eps 'single-float))
            (inv-n  (/ 1.0f0 (coerce n 'single-float))))
        (declare (type (simple-array single-float (*)) ad gd)
                 (type single-float eps-sf inv-n)
                 (type fixnum m n)
                 (optimize (speed 3) (safety 1)))
        (dotimes (i m)
          (let ((base (the fixnum (* i n))))
            ;; 1. Sum of squares across the row.
            (let ((sumsq 0.0f0))
              (declare (type single-float sumsq))
              (dotimes (j n)
                (let ((v (aref ad (the fixnum (+ base j)))))
                  (incf sumsq (* v v))))
              ;; 2. RMS + eps, then invert once so inner loop uses multiply.
              ;; The argument to sqrt is sum-of-squares (>=0) plus positive eps,
              ;; so it's always non-negative. Promise that so SBCL emits real sqrt.
              (let ((inv-rms (/ 1.0f0
                                (sqrt (the (single-float 0.0)
                                           (+ (* sumsq inv-n) eps-sf))))))
                (declare (type single-float inv-rms))
                ;; 3. Normalize and scale by gain.
                (dotimes (j n)
                  (setf (aref ad (the fixnum (+ base j)))
                        (* (aref ad (the fixnum (+ base j)))
                           inv-rms
                           (aref gd j)))))))))))
  a)

;;; ------------------------------------------------------------------
;;; GELU (elementwise, in-place)
;;; ------------------------------------------------------------------
;;; Tanh approximation (Hendrycks & Gimpel 2016), matches what GPT-2 uses:
;;;   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
;;;
;;; The "exact" GELU uses erf and is slightly slower to compute; the tanh
;;; form is accurate to a few decimals and standard in transformer code.
;;;
;;; Shape-agnostic: walks the flat buffer, ignores dimensionality.

(defconstant +gelu-c1+ (coerce (sqrt (/ 2.0d0 pi)) 'single-float)) ; ~0.7978846
(defconstant +gelu-c2+ 0.044715f0)

(defun gelu! (a)
  "In-place GELU on every element of A. Any shape."
  (let ((ad (tensor-data a))
        (c1 +gelu-c1+)
        (c2 +gelu-c2+))
    (declare (type (simple-array single-float (*)) ad)
             (type single-float c1 c2)
             (optimize (speed 3) (safety 1)))
    (dotimes (i (length ad))
      (let* ((x  (aref ad i))
             (x3 (* x x x))
             (inner (* c1 (+ x (* c2 x3))))
             (t-inner (the single-float (tanh inner))))
        (declare (type single-float x x3 inner t-inner))
        (setf (aref ad i) (* 0.5f0 x (+ 1.0f0 t-inner))))))
  a)

;;; ------------------------------------------------------------------
;;; Tests
;;; ------------------------------------------------------------------

(defun run-tensor-ops-2-tests ()
  (format t "~&Running tensor-ops-2 tests...~%")

  ;; --- bmatmul: 2 batches of (2x3) * (3x2) = (2x2) ---
  ;; Batch 0: same numbers as the matmul test in tensor-ops.lisp -> [[58 64][139 154]]
  ;; Batch 1: A = identity-ish [[1 0 0][0 1 0]] * B [[7 8][9 10][11 12]] = [[7 8][9 10]]
  (let* ((a (make-from-list '(2 2 3)
                            '(1 2 3 4 5 6      ; batch 0
                              1 0 0 0 1 0)))   ; batch 1
         (b (make-from-list '(2 3 2)
                            '(7 8 9 10 11 12   ; batch 0
                              7 8 9 10 11 12))); batch 1
         (c (bmatmul a b)))
    (assert (equal '(2 2 2) (tensor-shape c)))
    ;; Batch 0
    (assert (approx= 58.0  (tensor-ref c 0 0 0)))
    (assert (approx= 64.0  (tensor-ref c 0 0 1)))
    (assert (approx= 139.0 (tensor-ref c 0 1 0)))
    (assert (approx= 154.0 (tensor-ref c 0 1 1)))
    ;; Batch 1
    (assert (approx= 7.0   (tensor-ref c 1 0 0)))
    (assert (approx= 8.0   (tensor-ref c 1 0 1)))
    (assert (approx= 9.0   (tensor-ref c 1 1 0)))
    (assert (approx= 10.0  (tensor-ref c 1 1 1))))

  ;; --- rmsnorm!: hand-checked ---
  ;; Row [1 2 3], n=3, sumsq=14, mean=14/3, rms=sqrt(14/3+eps) ~= 2.1602
  ;; Output ~= [0.4629, 0.9258, 1.3887] with gain=[1 1 1]
  (let ((a    (make-from-list '(1 3) '(1 2 3)))
        (gain (make-from-list '(3)   '(1 1 1))))
    (rmsnorm! a gain)
    (assert (approx= 0.4629 (tensor-ref a 0 0) 1e-3))
    (assert (approx= 0.9258 (tensor-ref a 0 1) 1e-3))
    (assert (approx= 1.3887 (tensor-ref a 0 2) 1e-3)))

  ;; --- rmsnorm!: gain actually scales ---
  ;; Same input, gain=[2 2 2] -> output doubles.
  (let ((a    (make-from-list '(1 3) '(1 2 3)))
        (gain (make-from-list '(3)   '(2 2 2))))
    (rmsnorm! a gain)
    (assert (approx= (* 2 0.4629) (tensor-ref a 0 0) 1e-3))
    (assert (approx= (* 2 1.3887) (tensor-ref a 0 2) 1e-3)))

  ;; --- rmsnorm!: two rows treated independently ---
  ;; Row 0: [1 2 3] as above.
  ;; Row 1: [10 20 30] -> same normalized output as row 0 (scale invariant).
  (let ((a    (make-from-list '(2 3) '(1 2 3 10 20 30)))
        (gain (make-from-list '(3)   '(1 1 1))))
    (rmsnorm! a gain)
    (assert (approx= (tensor-ref a 0 0) (tensor-ref a 1 0) 1e-3))
    (assert (approx= (tensor-ref a 0 2) (tensor-ref a 1 2) 1e-3)))

  ;; --- gelu!: known values ---
  ;; gelu(0) = 0
  ;; gelu(1) ~= 0.8412
  ;; gelu(-1) ~= -0.1588
  ;; gelu(large positive) ~= x
  ;; gelu(large negative) ~= 0
  (let ((a (make-from-list '(5) '(0 1 -1 10 -10))))
    (gelu! a)
    (assert (approx= 0.0     (tensor-ref a 0) 1e-4))
    (assert (approx= 0.8412  (tensor-ref a 1) 1e-3))
    (assert (approx= -0.1588 (tensor-ref a 2) 1e-3))
    (assert (approx= 10.0    (tensor-ref a 3) 1e-3))
    (assert (approx= 0.0     (tensor-ref a 4) 1e-3)))

  (format t "All tensor-ops-2 tests passed.~%"))

(run-tensor-ops-2-tests)
