;;;; tensor-ops.lisp
;;;; Core tensor primitives for Raven-LLM.
;;;;
;;;; How to run:
;;;;   (load (compile-file "src/tensor-ops.lisp"))
;;;;   Tests run automatically at load time; look for "All tensor-ops tests passed."
;;;;
;;;; Design decisions:
;;;;   - Element type: single-float (32-bit). Standard for neural nets.
;;;;   - Storage: struct with flat (simple-array single-float (*)) + shape list.
;;;;     Shape travels with the data; access has one struct-slot deref of overhead.
;;;;   - Layout: row-major. Last axis varies fastest.
;;;;   - No external packages. Everything defined here.
;;;;
;;;; Note on (the fixnum ...) sprinkled through hot loops:
;;;;   SBCL knows loop counters are fixnums, but can't prove that fixnum*fixnum
;;;;   fits in a fixnum (worst-case overflow to bignum). The (the fixnum ...)
;;;;   promises the result fits, letting SBCL emit tight integer arithmetic
;;;;   instead of generic. For array indices bounded by array size, this is safe.

;;; ------------------------------------------------------------------
;;; Tensor struct and constructors
;;; ------------------------------------------------------------------

(defstruct tensor
  ;; Flat backing storage. Length = product of shape.
  (data  (make-array 0 :element-type 'single-float) :type (simple-array single-float (*)))
  ;; Shape as a list, e.g. (3 4) for a 3x4 matrix, (5) for a length-5 vector.
  (shape '()                                        :type list))

(defun shape-size (shape)
  "Total number of elements implied by SHAPE."
  (reduce #'* shape :initial-value 1))

(defun make-zeros (shape)
  "Allocate a tensor of the given SHAPE, initialized to 0.0."
  (let* ((n   (shape-size shape))
         (buf (make-array n :element-type 'single-float :initial-element 0.0f0)))
    (make-tensor :data buf :shape (copy-list shape))))

(defun make-from-list (shape values)
  "Build a tensor of SHAPE from a flat list of VALUES (row-major order).
   Convenient for tests and small hand-built matrices."
  (let* ((n (shape-size shape)))
    (assert (= n (length values)) ()
            "make-from-list: shape ~A needs ~D values, got ~D."
            shape n (length values))
    (let ((buf (make-array n :element-type 'single-float)))
      (loop for v in values
            for i from 0
            do (setf (aref buf i) (coerce v 'single-float)))
      (make-tensor :data buf :shape (copy-list shape)))))

;;; ------------------------------------------------------------------
;;; Indexing
;;; ------------------------------------------------------------------
;;; Row-major flat index for coords (i j k ...) with shape (d0 d1 d2 ...):
;;;   flat = i*(d1*d2*...) + j*(d2*...) + k*(...) + ...
;;; Computed by folding left-to-right, multiplying by each successive dim.

(defun flat-index (shape coords)
  "Convert a list of COORDS into a row-major flat index against SHAPE.
   Signals an error if arity or bounds are wrong."
  (assert (= (length shape) (length coords)) ()
          "flat-index: shape ~A and coords ~A have different arities." shape coords)
  (let ((idx 0))
    (loop for c    in coords
          for d    in shape
          do (assert (and (>= c 0) (< c d)) ()
                     "flat-index: coord ~D out of bounds for dim ~D." c d)
             (setf idx (+ (* idx d) c)))
    idx))

(defun tensor-ref (tensor &rest coords)
  "Read element at COORDS from TENSOR."
  (aref (tensor-data tensor) (flat-index (tensor-shape tensor) coords)))

(defun (setf tensor-ref) (value tensor &rest coords)
  "Write VALUE into TENSOR at COORDS."
  (setf (aref (tensor-data tensor) (flat-index (tensor-shape tensor) coords))
        (coerce value 'single-float)))

;;; ------------------------------------------------------------------
;;; Random initialization
;;; ------------------------------------------------------------------
;;; Uniform in [-scale, +scale]. Real init schemes (Xavier, He) come later;
;;; this is enough to get non-zero activations for testing.

(defun fill-random! (tensor &key (scale 0.1))
  "Fill TENSOR in-place with uniform random values in [-SCALE, +SCALE]."
  (let ((buf   (tensor-data tensor))
        (s     (coerce scale 'single-float))
        (two-s (coerce (* 2 scale) 'single-float)))
    (dotimes (i (length buf))
      (setf (aref buf i) (- (random two-s) s)))
    tensor))

;;; ------------------------------------------------------------------
;;; Matmul (2D x 2D -> 2D)
;;; ------------------------------------------------------------------
;;; A is (m k), B is (k n), C is (m n). C[i,j] = sum over p of A[i,p]*B[p,j].
;;; Naive triple loop with loop order i-p-j so B[p,*] and C[i,*] are walked
;;; contiguously in row-major memory.
;;; Row-base products hoisted out of inner loops (both for perf and to give
;;; SBCL a single named fixnum to reason about).

(defun matmul (a b)
  "Matrix-matrix multiply. A: (m k), B: (k n) -> result: (m n)."
  (let ((ashape (tensor-shape a))
        (bshape (tensor-shape b)))
    (assert (and (= 2 (length ashape)) (= 2 (length bshape))) ()
            "matmul: both args must be 2D, got shapes ~A and ~A." ashape bshape)
    (let ((m  (first  ashape))
          (k  (second ashape))
          (kb (first  bshape))
          (n  (second bshape)))
      (assert (= k kb) ()
              "matmul: inner dims disagree: ~A vs ~A." ashape bshape)
      (let* ((out (make-zeros (list m n)))
             (ad  (tensor-data a))
             (bd  (tensor-data b))
             (od  (tensor-data out)))
        (declare (type (simple-array single-float (*)) ad bd od)
                 (type fixnum m k n)
                 (optimize (speed 3) (safety 1)))
        (dotimes (i m)
          (let ((i-k (the fixnum (* i k)))
                (i-n (the fixnum (* i n))))
            (dotimes (p k)
              (let ((aip (aref ad (the fixnum (+ i-k p))))
                    (p-n (the fixnum (* p n))))
                (dotimes (j n)
                  (incf (aref od (the fixnum (+ i-n j)))
                        (* aip (aref bd (the fixnum (+ p-n j))))))))))
        out))))

;;; ------------------------------------------------------------------
;;; Elementwise add (in-place)
;;; ------------------------------------------------------------------

(defun add! (dest src)
  "Elementwise DEST += SRC, in place. Shapes must match exactly."
  (assert (equal (tensor-shape dest) (tensor-shape src)) ()
          "add!: shape mismatch ~A vs ~A."
          (tensor-shape dest) (tensor-shape src))
  (let ((dd (tensor-data dest))
        (sd (tensor-data src)))
    (declare (type (simple-array single-float (*)) dd sd)
             (optimize (speed 3) (safety 1)))
    (dotimes (i (length dd))
      (incf (aref dd i) (aref sd i))))
  dest)

;;; ------------------------------------------------------------------
;;; Broadcast-add: matrix + row-vector
;;; ------------------------------------------------------------------
;;; MAT is (m n), BIAS is (n). Adds BIAS to every row of MAT, in place.
;;; This is the shape you get adding a bias vector after a linear layer.

(defun broadcast-add! (mat bias)
  "In-place: add row-vector BIAS (n) to each row of matrix MAT (m n)."
  (let ((mshape (tensor-shape mat))
        (bshape (tensor-shape bias)))
    (assert (and (= 2 (length mshape)) (= 1 (length bshape))) ()
            "broadcast-add!: need 2D matrix and 1D bias, got ~A and ~A." mshape bshape)
    (let ((m  (first  mshape))
          (n  (second mshape))
          (nb (first  bshape)))
      (assert (= n nb) ()
              "broadcast-add!: bias length ~D must match matrix cols ~D." nb n)
      (let ((md (tensor-data mat))
            (bd (tensor-data bias)))
        (declare (type (simple-array single-float (*)) md bd)
                 (type fixnum m n)
                 (optimize (speed 3) (safety 1)))
        (dotimes (i m)
          (let ((row-base (the fixnum (* i n))))
            (dotimes (j n)
              (incf (aref md (the fixnum (+ row-base j))) (aref bd j))))))))
  mat)

;;; ------------------------------------------------------------------
;;; Transpose (2D)
;;; ------------------------------------------------------------------

(defun transpose (a)
  "Return a new tensor that is the transpose of the 2D tensor A."
  (let ((s (tensor-shape a)))
    (assert (= 2 (length s)) () "transpose: 2D only, got shape ~A." s)
    (let* ((m   (first s))
           (n   (second s))
           (out (make-zeros (list n m)))
           (ad  (tensor-data a))
           (od  (tensor-data out)))
      (declare (type (simple-array single-float (*)) ad od)
               (type fixnum m n)
               (optimize (speed 3) (safety 1)))
      (dotimes (i m)
        (let ((i-n (the fixnum (* i n))))
          (dotimes (j n)
            ;; out[j,i] = a[i,j]
            (setf (aref od (the fixnum (+ (the fixnum (* j m)) i)))
                  (aref ad (the fixnum (+ i-n j)))))))
      out)))

;;; ------------------------------------------------------------------
;;; Softmax along last axis (in-place)
;;; ------------------------------------------------------------------
;;; For 2D input (m n), softmax each row independently.
;;; Numerically stable: subtract row max before exp, so largest exp arg is 0.
;;;
;;; Type notes:
;;;   SUM is explicitly declared single-float, and (exp ...) is wrapped with
;;;   (the single-float ...). Otherwise SBCL widens SUM's type to a float
;;;   union and falls back to generic arithmetic in the inner loop.

(defun softmax! (a)
  "In-place row-wise softmax on 2D tensor A (m n)."
  (let ((s (tensor-shape a)))
    (assert (= 2 (length s)) () "softmax!: 2D only, got shape ~A." s)
    (let ((m  (first s))
          (n  (second s))
          (ad (tensor-data a)))
      (declare (type (simple-array single-float (*)) ad)
               (type fixnum m n)
               (optimize (speed 3) (safety 1)))
      (dotimes (i m)
        (let ((base (the fixnum (* i n))))
          ;; 1. Find row max.
          (let ((row-max (aref ad base)))
            (declare (type single-float row-max))
            (loop for j fixnum from 1 below n
                  do (let ((v (aref ad (the fixnum (+ base j)))))
                       (when (> v row-max) (setf row-max v))))
            ;; 2. Subtract max, exponentiate, accumulate sum.
            (let ((sum 0.0f0))
              (declare (type single-float sum))
              (dotimes (j n)
                (let ((e (the single-float
                              (exp (- (aref ad (the fixnum (+ base j)))
                                      row-max)))))
                  (setf (aref ad (the fixnum (+ base j))) e)
                  (incf sum e)))
              ;; 3. Divide by sum.
              (let ((inv (/ 1.0f0 sum)))
                (declare (type single-float inv))
                (dotimes (j n)
                  (setf (aref ad (the fixnum (+ base j)))
                        (* (aref ad (the fixnum (+ base j))) inv))))))))))
  a)

;;; ------------------------------------------------------------------
;;; Quick self-tests
;;; ------------------------------------------------------------------
;;; Run at load time to catch regressions immediately. Fail loudly.

(defun approx= (x y &optional (eps 1e-5))
  (< (abs (- x y)) eps))

(defun run-tensor-ops-tests ()
  (format t "~&Running tensor-ops tests...~%")

  ;; --- indexing round-trip ---
  (let ((t1 (make-from-list '(2 3) '(1 2 3
                                     4 5 6))))
    (assert (= 1.0 (tensor-ref t1 0 0)))
    (assert (= 6.0 (tensor-ref t1 1 2)))
    (setf (tensor-ref t1 1 1) 99)
    (assert (= 99.0 (tensor-ref t1 1 1))))

  ;; --- matmul: (2x3) * (3x2) = (2x2), hand-checked ---
  ;; A = [[1 2 3] [4 5 6]], B = [[7 8] [9 10] [11 12]]
  ;; C = [[1*7+2*9+3*11, 1*8+2*10+3*12] [4*7+5*9+6*11, 4*8+5*10+6*12]]
  ;;   = [[58 64] [139 154]]
  (let* ((a (make-from-list '(2 3) '(1 2 3 4 5 6)))
         (b (make-from-list '(3 2) '(7 8 9 10 11 12)))
         (c (matmul a b)))
    (assert (equal '(2 2) (tensor-shape c)))
    (assert (approx= 58.0  (tensor-ref c 0 0)))
    (assert (approx= 64.0  (tensor-ref c 0 1)))
    (assert (approx= 139.0 (tensor-ref c 1 0)))
    (assert (approx= 154.0 (tensor-ref c 1 1))))

  ;; --- broadcast-add ---
  (let ((m (make-from-list '(2 3) '(1 1 1 2 2 2)))
        (b (make-from-list '(3)   '(10 20 30))))
    (broadcast-add! m b)
    (assert (approx= 11.0 (tensor-ref m 0 0)))
    (assert (approx= 21.0 (tensor-ref m 0 1)))
    (assert (approx= 32.0 (tensor-ref m 1 2))))

  ;; --- transpose ---
  (let* ((a  (make-from-list '(2 3) '(1 2 3 4 5 6)))
         (at (transpose a)))
    (assert (equal '(3 2) (tensor-shape at)))
    (assert (= 1.0 (tensor-ref at 0 0)))
    (assert (= 4.0 (tensor-ref at 0 1)))
    (assert (= 6.0 (tensor-ref at 2 1))))

  ;; --- softmax: each row sums to 1, and equal inputs give equal outputs ---
  (let ((a (make-from-list '(2 3) '(1 1 1
                                    0 0 100))))
    (softmax! a)
    ;; Row 0: uniform -> all 1/3.
    (assert (approx= (/ 1.0 3.0) (tensor-ref a 0 0)))
    (assert (approx= (/ 1.0 3.0) (tensor-ref a 0 1)))
    (assert (approx= (/ 1.0 3.0) (tensor-ref a 0 2)))
    ;; Row 1: one huge value dominates, others ~0.
    (assert (approx= 0.0 (tensor-ref a 1 0)))
    (assert (approx= 0.0 (tensor-ref a 1 1)))
    (assert (approx= 1.0 (tensor-ref a 1 2)))
    ;; Rows sum to 1.
    (dotimes (i 2)
      (let ((s 0.0))
        (dotimes (j 3) (incf s (tensor-ref a i j)))
        (assert (approx= 1.0 s)))))

  (format t "All tensor-ops tests passed.~%"))

;; Auto-run tests when this file is loaded.
(run-tensor-ops-tests)
