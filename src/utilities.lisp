;;;; src/utilities.lisp
;;;;
;;;; Project-wide utilities used by multiple modules.
;;;;
;;;; Contents:
;;;;   - LOAD-CORPUS         : read a UTF-8 text file into a string, correctly
;;;;                           handling the case where char count < byte count.
;;;;   - Tensor constructors : MAKE-TENSOR, ZEROS, ONES, TENSOR-LIKE
;;;;                           Thin wrappers over MAKE-ARRAY that guarantee
;;;;                           SIMPLE-ARRAY of SINGLE-FLOAT so SBCL can
;;;;                           generate tight code (see note below).
;;;;   - Random init         : RANDN, XAVIER-INIT
;;;;                           For weight initialization; RANDN uses the
;;;;                           Box-Muller transform (no external packages).
;;;;   - Shape helpers       : TENSOR-SHAPE, TENSOR-NUMEL
;;;;
;;;; Numeric type: SINGLE-FLOAT throughout. Double-float would double memory
;;;; use for no accuracy benefit at this model size, and SBCL's single-float
;;;; math is well-optimized on modern hardware.
;;;;
;;;; Array type: (SIMPLE-ARRAY SINGLE-FLOAT (*)) or (SIMPLE-ARRAY SINGLE-FLOAT
;;;; (* *)) etc. SIMPLE-ARRAY (no fill-pointer, not displaced, not adjustable)
;;;; is the type SBCL can optimize aggressively. Every tensor in this project
;;;; is a simple-array of single-float. Do not pass around adjustable arrays
;;;; or arrays with fill pointers.
;;;;
;;;; How to run (from project root):
;;;;   (load (compile-file "src/utilities.lisp"))
;;;;   (defparameter *text* (load-corpus "corpus/poeall.txt"))
;;;;   (defparameter *w* (xavier-init 64 64))

;;; ---------- Corpus loading ----------
;;; The gotcha: (make-string (file-length stream)) allocates one character
;;; per BYTE in the file, but UTF-8 characters may be multiple bytes. So
;;; the buffer is oversized and read-sequence fills only part of it,
;;; leaving trailing NUL characters. Fix: use read-sequence's return value
;;; (the number of chars actually read) to trim.

(defun load-corpus (path)
  "Read entire file at PATH into a string. Handles multi-byte UTF-8
   correctly by trimming the buffer to the actual char count returned
   by READ-SEQUENCE."
  (with-open-file (stream path :direction :input)
    (let* ((buffer (make-string (file-length stream)))
           (actual (read-sequence buffer stream)))
      (subseq buffer 0 actual))))

;;; ---------- Tensor constructors ----------
;;; All tensors are (SIMPLE-ARRAY SINGLE-FLOAT (dims...)). MAKE-TENSOR is
;;; the base constructor; ZEROS/ONES/TENSOR-LIKE are convenience wrappers.
;;;
;;; DIMS is a list of positive integers (e.g. '(64 64) for a 64x64 matrix,
;;; '(64) for a length-64 vector). We use a list rather than &rest args so
;;; callers can build shapes programmatically without APPLY gymnastics.

(defun make-tensor (dims &key (initial-element 0.0f0))
  "Allocate a SIMPLE-ARRAY of SINGLE-FLOAT with shape DIMS.
   DIMS is a list, e.g. '(64 64). INITIAL-ELEMENT defaults to 0.0f0."
  (make-array dims
              :element-type 'single-float
              :initial-element (coerce initial-element 'single-float)))

(defun zeros (dims)
  "Tensor of shape DIMS filled with 0.0f0."
  (make-tensor dims :initial-element 0.0f0))

(defun ones (dims)
  "Tensor of shape DIMS filled with 1.0f0."
  (make-tensor dims :initial-element 1.0f0))

(defun tensor-like (tensor &key (initial-element 0.0f0))
  "New tensor with the same shape as TENSOR. Useful for gradient buffers."
  (make-tensor (array-dimensions tensor) :initial-element initial-element))

;;; ---------- Random initialization ----------
;;; Box-Muller transform: given two uniform(0,1) samples U1, U2,
;;;   Z0 = sqrt(-2 ln U1) * cos(2 pi U2)
;;; is a standard normal sample. We use one of the two outputs per call
;;; and discard the other for code simplicity.
;;;
;;; CL's RANDOM returns uniform in [0, arg). We guard against U1 = 0
;;; because log(0) = -inf.

(defun randn ()
  "Sample from the standard normal distribution N(0, 1). Returns SINGLE-FLOAT."
  (let ((u1 (max (random 1.0f0) 1.0f-7))   ; avoid log(0)
        (u2 (random 1.0f0)))
    (* (sqrt (* -2.0f0 (log u1)))
       (cos (* 2.0f0 (float pi 1.0f0) u2)))))

;;; Xavier (Glorot) initialization: weights drawn from N(0, 2/(fan_in + fan_out)).
;;; Standard choice for layers with symmetric activations. For a linear layer
;;; mapping fan_in -> fan_out, this keeps activation variance roughly constant
;;; across layers at initialization. The variance formula is symmetric in
;;; fan_in/fan_out (only their SUM matters), so which one is "in" vs "out"
;;; never changes the numbers — only which shape convention the caller wants.
;;;
;;; Shape returned: (D-IN D-OUT), matching THIS PROJECT's actual matmul
;;; convention (see modules.lisp header): activations are row vectors
;;; multiplied on the LEFT, y = x . W, so W must be stored (d_in, d_out) for
;;; MATMUL-FORWARD(x, W) to work with no transpose. (This is the opposite of
;;; the classical linear-algebra convention y = W . x with column vectors
;;; and W stored (fan_out, fan_in) — if you've seen Xavier init described
;;; that way elsewhere, that's why the axis order here looks flipped.)
;;; Every call site in this codebase passes (xavier-init d-in d-out) in that
;;; order — e.g. FFN's W1 is (xavier-init d-model d-ff), producing the (D F)
;;; shape FFN-FWD expects for INPUT (T D) . W1 -> (T F).

(defun xavier-init (d-in d-out)
  "Xavier-initialized weight matrix of shape (D-IN D-OUT) — ready to use
   directly as MATMUL-FORWARD's second argument against a (T D-IN)
   input, per this project's (d_in, d_out) weight-storage convention
   (see the file-section comment above). Samples from N(0, 2/(d_in + d_out))."
  (let* ((std (sqrt (/ 2.0f0 (+ d-in d-out))))
         (tensor (make-tensor (list d-in d-out))))
    (dotimes (i (array-total-size tensor))
      (setf (row-major-aref tensor i)
            (* std (randn))))
    tensor))

;;; ---------- Shape helpers ----------
;;; Thin wrappers that make call sites read better than raw ARRAY-DIMENSIONS
;;; and ARRAY-TOTAL-SIZE. Also give us a single place to add shape validation
;;; later if we want it.

(defun tensor-shape (tensor)
  "Return the shape of TENSOR as a list."
  (array-dimensions tensor))

(defun tensor-numel (tensor)
  "Total number of elements in TENSOR."
  (array-total-size tensor))
