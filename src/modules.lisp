;;;; src/modules.lisp
;;;;
;;;; Modules are the composable units the DSL wires together. Each module
;;;; type has three registered functions:
;;;;
;;;;   allocator : (config) -> params-plist
;;;;       Given the module's config (from the DSL), allocate and initialize
;;;;       its parameter tensors. Returns a plist like (:w #<tensor> :b ...).
;;;;
;;;;   forward   : (params input context) -> (values output saved)
;;;;       Runs the module forward. CONTEXT is a plist of things the module
;;;;       may need beyond its own params (e.g. current sequence length,
;;;;       causal-mask flag). SAVED is whatever the backward needs.
;;;;
;;;;   backward  : (params saved grad-out) -> (values grad-input grad-params)
;;;;       Returns gradient w.r.t. input and a plist of gradients w.r.t.
;;;;       each parameter (same keys as params).
;;;;
;;;; The registry is a hashtable from module-type keyword to a plist of
;;;; those three functions. REGISTER-MODULE adds an entry; GET-MODULE
;;;; retrieves one. The DSL/model layer calls into the registry — it never
;;;; hardcodes module names.
;;;;
;;;; Modules in this file:
;;;;   :embedding             token id -> vector
;;;;   :positional-sinusoidal add sinusoidal position encoding
;;;;   :rmsnorm               root-mean-square normalization with learned scale
;;;;   :attention-block       multi-head causal self-attention
;;;;   :ffn                   two-layer MLP with GELU
;;;;   :transformer-block     rmsnorm -> attention -> residual ->
;;;;                          rmsnorm -> ffn -> residual (pre-norm variant)
;;;;   :unembedding           final projection to vocab logits (tied option)
;;;;
;;;; Shape convention throughout: activations are (T d_model) where T is
;;;; the current sequence length. Batching is one sequence at a time — no
;;;; batch dimension. Fine at this model scale.
;;;;
;;;; How to run (from project root):
;;;;   (load (compile-file "src/utilities.lisp"))
;;;;   (load (compile-file "src/tensor-ops.lisp"))
;;;;   (load (compile-file "src/modules.lisp"))


;;; ==========================================================================
;;; Module registry
;;; ==========================================================================

(defvar *module-registry* (make-hash-table :test 'eq)
  "Maps module-type keyword -> plist (:allocator :forward :backward).")

(defun register-module (type &key allocator forward backward)
  "Register a module type. TYPE is a keyword; the three fns are functions."
  (setf (gethash type *module-registry*)
        (list :allocator allocator :forward forward :backward backward)))

(defun get-module (type)
  "Return the plist for TYPE, or error if not registered."
  (or (gethash type *module-registry*)
      (error "Unknown module type: ~S" type)))

(defun module-alloc (type config)
  (funcall (getf (get-module type) :allocator) config))

(defun module-forward (type params input context)
  (funcall (getf (get-module type) :forward) params input context))

(defun module-backward (type params saved grad-out)
  (funcall (getf (get-module type) :backward) params saved grad-out))


;;; ==========================================================================
;;; :embedding    token id -> row of learned table
;;; ==========================================================================
;;;
;;; Config: (:vocab-size N :d-model D)
;;; Params: (:table (V D) tensor)
;;; Input:  (simple-array fixnum (T))  -- token ids
;;; Output: (T D) tensor
;;;
;;; Init: N(0, 1) scaled by 1/sqrt(d) is a common choice; here we use
;;; xavier-init treating the table as a (V D) matrix. Not the "correct"
;;; theoretical init for an embedding but works fine.

(defun embedding-alloc (config)
  (let ((v (getf config :vocab-size))
        (d (getf config :d-model)))
    (list :table (xavier-init v d))))

(defun embedding-fwd (params input context)
  (declare (ignore context))
  (embedding-lookup-forward (getf params :table) input))

(defun embedding-bwd (params saved grad-out)
  (declare (ignore params))
  ;; No gradient w.r.t. input (ids aren't differentiable).
  (values nil (list :table (embedding-lookup-backward grad-out saved))))

(register-module :embedding
                 :allocator #'embedding-alloc
                 :forward   #'embedding-fwd
                 :backward  #'embedding-bwd)


;;; ==========================================================================
;;; :positional-sinusoidal    add fixed sin/cos position encoding
;;; ==========================================================================
;;;
;;; Config: (:d-model D :max-len L)
;;; Params: none (encoding is deterministic)
;;; Input:  (T D) tensor
;;; Output: (T D) tensor  = input + PE[:T, :]
;;;
;;; PE[pos, 2i]   = sin(pos / 10000^(2i/D))
;;; PE[pos, 2i+1] = cos(pos / 10000^(2i/D))
;;;
;;; We precompute the PE table once at allocation time (up to max-len) and
;;; stash it in the params plist under :pe — treated as a constant, no
;;; gradient flows to it. This is cheaper than recomputing every forward.
;;;
;;; Backward: grad_input = grad_out (PE is additive constant). Nothing else.

(defun build-sinusoidal-pe (max-len d)
  "Precompute the (MAX-LEN D) sinusoidal position table."
  (let ((pe (make-tensor (list max-len d))))
    (declare (type tensor-2d pe))
    (dotimes (pos max-len)
      (dotimes (i d)
        (let* ((two-i (* 2 (floor i 2)))     ; 0,0,2,2,4,4,... (pair index)
               (freq  (expt 10000.0f0 (/ (float two-i 1.0f0)
                                         (float d      1.0f0))))
               (angle (/ (float pos 1.0f0) freq)))
          (setf (aref pe pos i)
                (if (evenp i) (sin angle) (cos angle))))))
    pe))

(defun positional-sinusoidal-alloc (config)
  (let ((d (getf config :d-model))
        (l (getf config :max-len)))
    (list :pe (build-sinusoidal-pe l d))))

(defun positional-sinusoidal-fwd (params input context)
  (declare (ignore context))
  (let* ((pe    (getf params :pe))
         (t-len (array-dimension input 0))
         (d     (array-dimension input 1))
         (y     (tensor-like input)))
    (declare (type tensor-2d pe input y)
             (type fixnum t-len d))
    (dotimes (tt t-len)
      (dotimes (k d)
        (setf (aref y tt k) (+ (aref input tt k) (aref pe tt k)))))
    (values y nil)))

(defun positional-sinusoidal-bwd (params saved grad-out)
  (declare (ignore params saved))
  ;; grad_input = grad_out (fresh copy — caller convention).
  (let ((gi (tensor-like grad-out)))
    (dotimes (i (array-total-size grad-out))
      (setf (row-major-aref gi i) (row-major-aref grad-out i)))
    (values gi nil)))

(register-module :positional-sinusoidal
                 :allocator #'positional-sinusoidal-alloc
                 :forward   #'positional-sinusoidal-fwd
                 :backward  #'positional-sinusoidal-bwd)


;;; ==========================================================================
;;; :rmsnorm    y[t,k] = (x[t,k] / rms(x[t,:])) * gamma[k]
;;; ==========================================================================
;;;
;;; Config: (:d-model D :eps 1e-5)
;;; Params: (:gamma (D) tensor, ones-initialized)
;;; Input:  (T D)
;;; Output: (T D)
;;;
;;; rms(x[t,:]) = sqrt(mean(x[t,:]^2) + eps)
;;;
;;; Backward for one row (dropping the t subscript):
;;;   let ms  = mean(x^2) + eps
;;;   let rms = sqrt(ms)
;;;   let inv = 1 / rms
;;;   y[k]    = x[k] * inv * gamma[k]
;;;
;;;   d y[k] / d x[j] = gamma[k] * (delta_kj * inv + x[k] * d inv / d x[j])
;;;   d inv / d x[j]  = -0.5 * ms^(-3/2) * (2 x[j] / D)
;;;                   = -x[j] / (D * ms * rms) = -x[j] * inv / (D * ms)
;;;
;;;   grad_x[j] = sum_k grad_out[k] * gamma[k] * (delta_kj * inv - x[k] * x[j] * inv / (D * ms))
;;;             = inv * (grad_out[j] * gamma[j]
;;;                      - x[j] / (D * ms) * sum_k grad_out[k] * gamma[k] * x[k])
;;;
;;;   grad_gamma[k] += grad_out[k] * x[k] * inv     (summed across all rows)
;;;
;;; Save: (x inv-per-row ms-per-row) so we don't recompute in backward.

(defun rmsnorm-alloc (config)
  (let ((d (getf config :d-model)))
    (list :gamma (ones (list d))
          :eps   (coerce (or (getf config :eps) 1.0f-5) 'single-float))))

(defun rmsnorm-fwd (params input context)
  (declare (ignore context))
  (let* ((gamma (getf params :gamma))
         (eps   (getf params :eps))
         (t-len (array-dimension input 0))
         (d     (array-dimension input 1))
         (y     (tensor-like input))
         (invs  (make-tensor (list t-len)))
         (mss   (make-tensor (list t-len))))
    (declare (type tensor-2d input y)
             (type tensor-1d gamma invs mss)
             (type single-float eps)
             (type fixnum t-len d))
    (dotimes (tt t-len)
      (let ((sum-sq 0.0f0))
        (declare (type single-float sum-sq))
        (dotimes (k d)
          (let ((v (aref input tt k)))
            (declare (type single-float v))
            (incf sum-sq (* v v))))
        (let* ((ms  (+ (/ sum-sq (float d 1.0f0)) eps))
               (inv (/ 1.0f0 (sqrt ms))))
          (declare (type single-float ms inv))
          (setf (aref mss  tt) ms
                (aref invs tt) inv)
          (dotimes (k d)
            (setf (aref y tt k)
                  (* (aref input tt k) inv (aref gamma k)))))))
    (values y (list input invs mss))))

(defun rmsnorm-bwd (params saved grad-out)
  (destructuring-bind (input invs mss) saved
    (declare (type tensor-2d input grad-out)
             (type tensor-1d invs mss))
    (let* ((gamma (getf params :gamma))
           (t-len (array-dimension input 0))
           (d     (array-dimension input 1))
           (grad-input (tensor-like input))
           (grad-gamma (make-tensor (list d))))
      (declare (type tensor-1d gamma grad-gamma)
               (type tensor-2d grad-input)
               (type fixnum t-len d))
      (dotimes (tt t-len)
        (let* ((inv (aref invs tt))
               (ms  (aref mss  tt))
               ;; s = sum_k grad_out[k] * gamma[k] * x[k]
               (s   0.0f0))
          (declare (type single-float inv ms s))
          (dotimes (k d)
            (incf s (* (aref grad-out tt k) (aref gamma k) (aref input tt k))))
          (let ((coef (/ (* inv s) (* (float d 1.0f0) ms))))
            (declare (type single-float coef))
            (dotimes (j d)
              (setf (aref grad-input tt j)
                    (- (* inv (aref grad-out tt j) (aref gamma j))
                       (* coef (aref input tt j))))))
          ;; accumulate grad_gamma
          (dotimes (k d)
            (incf (aref grad-gamma k)
                  (* (aref grad-out tt k) (aref input tt k) inv)))))
      (values grad-input (list :gamma grad-gamma)))))

(register-module :rmsnorm
                 :allocator #'rmsnorm-alloc
                 :forward   #'rmsnorm-fwd
                 :backward  #'rmsnorm-bwd)


;;; ==========================================================================
;;; :attention-block    multi-head causal self-attention
;;; ==========================================================================
;;;
;;; Config: (:d-model D :n-heads H)   -- D must be divisible by H
;;; Params: (:wq (D D) :wk (D D) :wv (D D) :wo (D D))
;;;         (Biases omitted; modern LLMs typically drop attention biases.)
;;; Input:  (T D)
;;; Output: (T D)
;;;
;;; Steps for each head h (dk = D / H):
;;;   1. Project: Q_h = X . Wq_h,  K_h = X . Wk_h,  V_h = X . Wv_h
;;;      (Wq_h is columns [h*dk : (h+1)*dk] of Wq; we slice on the fly.)
;;;   2. Scores: S_h = Q_h . K_h^T  * (1/sqrt(dk))
;;;   3. Causal mask: S_h[i,j] = -inf for j > i
;;;   4. Softmax rows: A_h = softmax(S_h)
;;;   5. Weighted V: Z_h = A_h . V_h              shape (T dk)
;;; Concat heads across the D axis -> Z (T D)
;;; Output: Y = Z . Wo
;;;
;;; This is written as a monolithic forward/backward for clarity. Speed
;;; optimizations (batched matmul across heads, fused ops) are deferred.
;;;
;;; The backward derivation is standard multi-head attention backprop.
;;; See e.g. https://arxiv.org/abs/1706.03762 for forward; backward is
;;; the routine chain-rule composition through matmul + softmax.
;;;
;;; NOTE: implementation is fairly long. Kept in one function to keep
;;; the derivation traceable end-to-end. Consider factoring per-head into
;;; a helper later if it stays stable.

(defun attention-alloc (config)
  (let ((d (getf config :d-model))
        (h (getf config :n-heads)))
    (assert (zerop (mod d h)) () "d-model ~A not divisible by n-heads ~A" d h)
    (list :wq (xavier-init d d)
          :wk (xavier-init d d)
          :wv (xavier-init d d)
          :wo (xavier-init d d)
          :n-heads h)))

(defun attention-fwd (params input context)
  (declare (ignore context))
  (let* ((wq (getf params :wq))
         (wk (getf params :wk))
         (wv (getf params :wv))
         (wo (getf params :wo))
         (h  (getf params :n-heads))
         (t-len (array-dimension input 0))
         (d     (array-dimension input 1))
         (dk    (floor d h))
         (scale-factor (/ 1.0f0 (sqrt (float dk 1.0f0)))))
    (declare (type tensor-2d input wq wk wv wo)
             (type fixnum t-len d dk h)
             (type single-float scale-factor))
    ;; Project Q, K, V by full-width matmul (T,D) @ (D,D) = (T,D).
    ;; NOTE: our matmul is y = A . B with A=(M,K), B=(K,N). Here A=input
    ;; (T,D), B=W (D,D), result (T,D). But WQ is stored as (D,D) with the
    ;; XAVIER-INIT convention (fan_out, fan_in) = (out_dim, in_dim). We
    ;; want out = in . W, treating W's first axis as input dim. So we
    ;; transpose W first. (Alternative: store W as (in,out) from the
    ;; start. Choosing transpose here to keep xavier-init's convention.)
    (multiple-value-bind (wqT _1) (transpose-forward wq) (declare (ignore _1))
      (multiple-value-bind (wkT _2) (transpose-forward wk) (declare (ignore _2))
        (multiple-value-bind (wvT _3) (transpose-forward wv) (declare (ignore _3))
          (multiple-value-bind (woT _4) (transpose-forward wo) (declare (ignore _4))
            (multiple-value-bind (q _q) (matmul-forward input wqT) (declare (ignore _q))
              (multiple-value-bind (k _k) (matmul-forward input wkT) (declare (ignore _k))
                (multiple-value-bind (v _v) (matmul-forward input wvT) (declare (ignore _v))
                  (declare (type tensor-2d q k v))
                  ;; Allocate output Z and per-head attention weights (for backward).
                  (let ((z          (make-tensor (list t-len d)))
                        (attn-all   (make-array h)))   ; vector of (T,T) tensors
                    (declare (type tensor-2d z))
                    (dotimes (hi h)
                      (let ((qh (make-tensor (list t-len dk)))
                            (kh (make-tensor (list t-len dk)))
                            (vh (make-tensor (list t-len dk))))
                        (declare (type tensor-2d qh kh vh))
                        ;; Slice head hi out of Q, K, V (columns [hi*dk : (hi+1)*dk]).
                        (dotimes (tt t-len)
                          (dotimes (kk dk)
                            (setf (aref qh tt kk) (aref q tt (+ (* hi dk) kk)))
                            (setf (aref kh tt kk) (aref k tt (+ (* hi dk) kk)))
                            (setf (aref vh tt kk) (aref v tt (+ (* hi dk) kk)))))
                        ;; Scores S = Q . K^T, then scale, then causal mask.
                        (multiple-value-bind (khT _kt) (transpose-forward kh) (declare (ignore _kt))
                          (multiple-value-bind (s _s) (matmul-forward qh khT) (declare (ignore _s))
                            (declare (type tensor-2d s))
                            (dotimes (i t-len)
                              (dotimes (j t-len)
                                (setf (aref s i j)
                                      (if (> j i)
                                          most-negative-single-float
                                          (* scale-factor (aref s i j))))))
                            (multiple-value-bind (a _a) (softmax-forward s) (declare (ignore _a))
                              (declare (type tensor-2d a))
                              (setf (aref attn-all hi) a)
                              ;; Zh = A . Vh, (T,T) @ (T,dk) = (T,dk)
                              (multiple-value-bind (zh _z) (matmul-forward a vh) (declare (ignore _z))
                                (declare (type tensor-2d zh))
                                (dotimes (tt t-len)
                                  (dotimes (kk dk)
                                    (setf (aref z tt (+ (* hi dk) kk))
                                          (aref zh tt kk))))))))))
                    ;; Y = Z . Wo^T
                    (multiple-value-bind (y _y) (matmul-forward z woT) (declare (ignore _y))
                      ;; SAVED bundles everything backward needs.
                      (values y (list :input input :q q :k k :v v :z z
                                      :attn attn-all
                                      :wq wq :wk wk :wv wv :wo wo
                                      :h h :dk dk :scale scale-factor)))))))))))))

(defun attention-bwd (params saved grad-out)
  (declare (ignore params))
  (let* ((input (getf saved :input))
         (q     (getf saved :q))
         (k     (getf saved :k))
         (v     (getf saved :v))
         (z     (getf saved :z))
         (attn  (getf saved :attn))
         (wq    (getf saved :wq))
         (wk    (getf saved :wk))
         (wv    (getf saved :wv))
         (wo    (getf saved :wo))
         (h     (getf saved :h))
         (dk    (getf saved :dk))
         (scale-factor (getf saved :scale))
         (t-len (array-dimension input 0))
         (d     (array-dimension input 1))
         (grad-input (tensor-like input))
         (grad-wq (tensor-like wq))
         (grad-wk (tensor-like wk))
         (grad-wv (tensor-like wv))
         (grad-wo (tensor-like wo)))
    (declare (type tensor-2d input q k v z wq wk wv wo grad-input grad-wq grad-wk grad-wv grad-wo grad-out)
             (type fixnum t-len d dk h)
             (type single-float scale-factor))
    ;; Reverse of: y = z . wo^T
    ;;   grad_z  = grad_y . wo         (grad_y is grad-out)
    ;;   grad_wo = grad_y^T . z, then transposed back  (see below)
    ;; We used wo^T in forward, so gradient w.r.t. wo^T is (grad_y^T . z)^T = z^T . grad_y.
    ;; Equivalently grad_wo = (grad_y^T . z) transposed, but easier: use
    ;; matmul-backward on the (z, wo^T) pair.
    (multiple-value-bind (woT _wot) (transpose-forward wo) (declare (ignore _wot))
      (multiple-value-bind (grad-z grad-woT)
          (matmul-backward grad-out (list z woT))
        (declare (type tensor-2d grad-z grad-woT))
        ;; grad-wo = transpose(grad-woT)
        (multiple-value-bind (gwo _tw) (transpose-forward grad-woT) (declare (ignore _tw))
          (dotimes (i (array-total-size wo))
            (setf (row-major-aref grad-wo i) (row-major-aref gwo i))))
        ;; Prepare accumulators for grad-Q, grad-K, grad-V (full width).
        (let ((grad-q (make-tensor (list t-len d)))
              (grad-k (make-tensor (list t-len d)))
              (grad-v (make-tensor (list t-len d))))
          (declare (type tensor-2d grad-q grad-k grad-v))
          (dotimes (hi h)
            (let ((qh (make-tensor (list t-len dk)))
                  (kh (make-tensor (list t-len dk)))
                  (vh (make-tensor (list t-len dk)))
                  (grad-zh (make-tensor (list t-len dk)))
                  (a  (aref attn hi)))
              (declare (type tensor-2d qh kh vh grad-zh a))
              ;; Slice heads and slice grad-z.
              (dotimes (tt t-len)
                (dotimes (kk dk)
                  (setf (aref qh tt kk) (aref q tt (+ (* hi dk) kk)))
                  (setf (aref kh tt kk) (aref k tt (+ (* hi dk) kk)))
                  (setf (aref vh tt kk) (aref v tt (+ (* hi dk) kk)))
                  (setf (aref grad-zh tt kk) (aref grad-z tt (+ (* hi dk) kk)))))
              ;; Zh = A . Vh; backward:
              (multiple-value-bind (grad-a grad-vh)
                  (matmul-backward grad-zh (list a vh))
                (declare (type tensor-2d grad-a grad-vh))
                ;; A = softmax(S_masked); backward:
                (let ((grad-s (softmax-backward grad-a a)))
                  (declare (type tensor-2d grad-s))
                  ;; Masked positions had -inf -> softmax=0 -> grad passes as 0 there
                  ;; already because A[i,j]=0 for masked j; but we also want to
                  ;; not push gradient into future positions. Zero them explicitly.
                  (dotimes (i t-len)
                    (dotimes (j t-len)
                      (when (> j i)
                        (setf (aref grad-s i j) 0.0f0))))
                  ;; S = scale * Qh . Kh^T; backward:
                  ;;   grad_qh_khT_result = scale * grad_s (chain through scale)
                  (dotimes (i (array-total-size grad-s))
                    (setf (row-major-aref grad-s i)
                          (* scale-factor (row-major-aref grad-s i))))
                  (multiple-value-bind (khT _kt) (transpose-forward kh) (declare (ignore _kt))
                    (multiple-value-bind (grad-qh grad-khT)
                        (matmul-backward grad-s (list qh khT))
                      (declare (type tensor-2d grad-qh grad-khT))
                      (multiple-value-bind (grad-kh _t2) (transpose-forward grad-khT) (declare (ignore _t2))
                        (declare (type tensor-2d grad-kh))
                        ;; Scatter head gradients back into the full (T,D) tensors.
                        (dotimes (tt t-len)
                          (dotimes (kk dk)
                            (setf (aref grad-q tt (+ (* hi dk) kk)) (aref grad-qh tt kk))
                            (setf (aref grad-k tt (+ (* hi dk) kk)) (aref grad-kh tt kk))
                            (setf (aref grad-v tt (+ (* hi dk) kk)) (aref grad-vh tt kk)))))))))))
          ;; Now propagate grad-Q, grad-K, grad-V back through their projections.
          ;; Forward was: Q = input . Wq^T, etc.
          (multiple-value-bind (wqT _1) (transpose-forward wq) (declare (ignore _1))
            (multiple-value-bind (wkT _2) (transpose-forward wk) (declare (ignore _2))
              (multiple-value-bind (wvT _3) (transpose-forward wv) (declare (ignore _3))
                (multiple-value-bind (gi-q gwqT) (matmul-backward grad-q (list input wqT))
                  (multiple-value-bind (gi-k gwkT) (matmul-backward grad-k (list input wkT))
                    (multiple-value-bind (gi-v gwvT) (matmul-backward grad-v (list input wvT))
                      (declare (type tensor-2d gi-q gi-k gi-v gwqT gwkT gwvT))
                      ;; grad_input = sum of three input-gradient contributions
                      (dotimes (i (array-total-size input))
                        (setf (row-major-aref grad-input i)
                              (+ (row-major-aref gi-q i)
                                 (row-major-aref gi-k i)
                                 (row-major-aref gi-v i))))
                      ;; grad_wq = transpose(gwqT), etc.
                      (multiple-value-bind (gwq _a) (transpose-forward gwqT) (declare (ignore _a))
                        (dotimes (i (array-total-size wq))
                          (setf (row-major-aref grad-wq i) (row-major-aref gwq i))))
                      (multiple-value-bind (gwk _b) (transpose-forward gwkT) (declare (ignore _b))
                        (dotimes (i (array-total-size wk))
                          (setf (row-major-aref grad-wk i) (row-major-aref gwk i))))
                      (multiple-value-bind (gwv _c) (transpose-forward gwvT) (declare (ignore _c))
                        (dotimes (i (array-total-size wv))
                          (setf (row-major-aref grad-wv i) (row-major-aref gwv i)))))))))))
          (values grad-input
                  (list :wq grad-wq :wk grad-wk :wv grad-wv :wo grad-wo)))))))

(register-module :attention-block
                 :allocator #'attention-alloc
                 :forward   #'attention-fwd
                 :backward  #'attention-bwd)


;;; ==========================================================================
;;; :ffn    two-layer MLP with GELU activation
;;; ==========================================================================
;;;
;;; Config: (:d-model D :d-ff F)   where F is usually 4*D
;;; Params: (:w1 (D F) :b1 (F) :w2 (F D) :b2 (D))
;;; Input:  (T D)
;;; Output: (T D)
;;;
;;;   h1 = input . w1 + b1        (T F)
;;;   h2 = gelu(h1)               (T F)
;;;   y  = h2 . w2 + b2           (T D)
;;;
;;; Convention note: unlike attention (where we stored W as (out,in) and
;;; transposed on use), here w1 and w2 are stored as (in,out) to match
;;; the matmul call sites directly. Xavier-init still takes (fan_out,
;;; fan_in) — so we call it with args flipped and get a matrix of shape
;;; (in, out).

(defun ffn-alloc (config)
  (let ((d (getf config :d-model))
        (f (or (getf config :d-ff) (* 4 (getf config :d-model)))))
    ;; xavier-init returns (fan-out fan-in); we want (fan-in fan-out),
    ;; so call with (fan-in fan-out) args to get shape (fan-in fan-out).
    ;; That's a swap: xavier-init D F -> (D F).
    (list :w1 (xavier-init d f)      ; shape (D F)
          :b1 (zeros (list f))
          :w2 (xavier-init f d)      ; shape (F D)
          :b2 (zeros (list d)))))

(defun ffn-fwd (params input context)
  (declare (ignore context))
  (let ((w1 (getf params :w1))
        (b1 (getf params :b1))
        (w2 (getf params :w2))
        (b2 (getf params :b2)))
    (multiple-value-bind (h1a _1) (matmul-forward input w1) (declare (ignore _1))
      (multiple-value-bind (h1 _2) (add-bias-forward h1a b1) (declare (ignore _2))
        (multiple-value-bind (h2 gelu-saved) (gelu-forward h1)
          (multiple-value-bind (h3 _3) (matmul-forward h2 w2) (declare (ignore _3))
            (multiple-value-bind (y _4) (add-bias-forward h3 b2) (declare (ignore _4))
              (values y (list :input input :h1a h1a :h1 h1 :h2 h2 :h3 h3
                              :w1 w1 :w2 w2 :gelu-saved gelu-saved)))))))))

(defun ffn-bwd (params saved grad-out)
  (declare (ignore params))
  (let ((input (getf saved :input))
        (h1a   (getf saved :h1a))
        (h1    (getf saved :h1))
        (h2    (getf saved :h2))
        (h3    (getf saved :h3))
        (w1    (getf saved :w1))
        (w2    (getf saved :w2))
        (gs    (getf saved :gelu-saved)))
    ;; y = h3 + b2  ->  grad_h3, grad_b2
    (multiple-value-bind (grad-h3 grad-b2) (add-bias-backward grad-out nil)
      (declare (ignore))
      ;; h3 = h2 . w2  ->  grad_h2, grad_w2
      (multiple-value-bind (grad-h2 grad-w2) (matmul-backward grad-h3 (list h2 w2))
        ;; h2 = gelu(h1) -> grad_h1
        (let ((grad-h1 (gelu-backward grad-h2 gs)))
          ;; h1 = h1a + b1 -> grad_h1a, grad_b1
          (multiple-value-bind (grad-h1a grad-b1) (add-bias-backward grad-h1 nil)
            (declare (ignore))
            ;; h1a = input . w1  ->  grad_input, grad_w1
            (multiple-value-bind (grad-input grad-w1)
                (matmul-backward grad-h1a (list input w1))
              (values grad-input
                      (list :w1 grad-w1 :b1 grad-b1
                            :w2 grad-w2 :b2 grad-b2)))))))))

(register-module :ffn
                 :allocator #'ffn-alloc
                 :forward   #'ffn-fwd
                 :backward  #'ffn-bwd)


;;; ==========================================================================
;;; :transformer-block    pre-norm variant
;;; ==========================================================================
;;;
;;; Config: (:d-model D :n-heads H :d-ff F :eps E)
;;; Params: (:norm1 <rmsnorm params> :attn <attention params>
;;;          :norm2 <rmsnorm params> :ffn  <ffn params>)
;;;
;;; Forward (pre-norm):
;;;   a  = attention(rmsnorm(x))
;;;   x1 = x + a               (residual)
;;;   f  = ffn(rmsnorm(x1))
;;;   y  = x1 + f              (residual)
;;;
;;; Pre-norm (normalize before sublayer, add residual after) is what modern
;;; LLMs use — it trains more stably at depth than post-norm.
;;;
;;; Backward: reverse the composition. Residual adds mean grad flows through
;;; both branches unchanged.

(defun transformer-block-alloc (config)
  (list :norm1 (module-alloc :rmsnorm         config)
        :attn  (module-alloc :attention-block config)
        :norm2 (module-alloc :rmsnorm         config)
        :ffn   (module-alloc :ffn             config)))

(defun transformer-block-fwd (params input context)
  (multiple-value-bind (n1  sn1) (module-forward :rmsnorm         (getf params :norm1) input context)
    (multiple-value-bind (a   sa)  (module-forward :attention-block (getf params :attn)  n1 context)
      ;; x1 = input + a
      (multiple-value-bind (x1 sadd1) (add-forward input a)
        (declare (ignore sadd1))
        (multiple-value-bind (n2 sn2) (module-forward :rmsnorm (getf params :norm2) x1 context)
          (multiple-value-bind (f sf) (module-forward :ffn (getf params :ffn) n2 context)
            (multiple-value-bind (y sadd2) (add-forward x1 f)
              (declare (ignore sadd2))
              (values y (list :sn1 sn1 :sa sa :sn2 sn2 :sf sf
                              :x1 x1)))))))))

(defun transformer-block-bwd (params saved grad-out)
  (let ((sn1 (getf saved :sn1))
        (sa  (getf saved :sa))
        (sn2 (getf saved :sn2))
        (sf  (getf saved :sf)))
    ;; y = x1 + f  ->  grad_x1_a, grad_f = grad_out both
    (multiple-value-bind (grad-x1-a grad-f) (add-backward grad-out nil)
      ;; f = ffn(n2) -> grad_n2, grad_ffn_params
      (multiple-value-bind (grad-n2 grad-ffn) (module-backward :ffn (getf params :ffn) sf grad-f)
        ;; n2 = rmsnorm(x1) -> grad_x1_b, grad_norm2_params
        (multiple-value-bind (grad-x1-b grad-n2p) (module-backward :rmsnorm (getf params :norm2) sn2 grad-n2)
          ;; grad_x1 total = grad_x1_a + grad_x1_b
          (let ((grad-x1 (tensor-like grad-x1-a)))
            (dotimes (i (array-total-size grad-x1))
              (setf (row-major-aref grad-x1 i)
                    (+ (row-major-aref grad-x1-a i)
                       (row-major-aref grad-x1-b i))))
            ;; x1 = input + a -> grad_input_a, grad_a
            (multiple-value-bind (grad-input-a grad-a) (add-backward grad-x1 nil)
              ;; a = attention(n1) -> grad_n1, grad_attn_params
              (multiple-value-bind (grad-n1 grad-attn) (module-backward :attention-block (getf params :attn) sa grad-a)
                ;; n1 = rmsnorm(input) -> grad_input_b, grad_norm1_params
                (multiple-value-bind (grad-input-b grad-n1p) (module-backward :rmsnorm (getf params :norm1) sn1 grad-n1)
                  (let ((grad-input (tensor-like grad-input-a)))
                    (dotimes (i (array-total-size grad-input))
                      (setf (row-major-aref grad-input i)
                            (+ (row-major-aref grad-input-a i)
                               (row-major-aref grad-input-b i))))
                    (values grad-input
                            (list :norm1 grad-n1p
                                  :attn  grad-attn
                                  :norm2 grad-n2p
                                  :ffn   grad-ffn))))))))))))

(register-module :transformer-block
                 :allocator #'transformer-block-alloc
                 :forward   #'transformer-block-fwd
                 :backward  #'transformer-block-bwd)


;;; ==========================================================================
;;; :unembedding    project (T D) -> (T V) logits
;;; ==========================================================================
;;;
;;; Config: (:d-model D :vocab-size V :tied nil-or-embedding-table)
;;; Params: (:w (D V))  -- or empty if tied
;;; Input:  (T D)
;;; Output: (T V) logits (no softmax; loss layer handles that)
;;;
;;; Weight tying: modern LLMs often share the embedding table with the
;;; unembedding projection. Since embedding is (V D) and unembedding wants
;;; (D V), tying means unembedding uses the embedding table transposed.
;;; If :tied is a tensor, we use it directly (as (V D)) and transpose at
;;; forward time. If :tied is NIL, we allocate our own (D V) weight.
;;;
;;; For now: no tying — allocate own weight. Add tying later if desired
;;; (requires a way for two modules to share a param, which the DSL layer
;;; will need to support).

(defun unembedding-alloc (config)
  (let ((d (getf config :d-model))
        (v (getf config :vocab-size)))
    (list :w (xavier-init d v))))    ; shape (D V)

(defun unembedding-fwd (params input context)
  (declare (ignore context))
  (matmul-forward input (getf params :w)))

(defun unembedding-bwd (params saved grad-out)
  (multiple-value-bind (grad-input grad-w) (matmul-backward grad-out saved)
    (declare (ignore params))
    (values grad-input (list :w grad-w))))

(register-module :unembedding
                 :allocator #'unembedding-alloc
                 :forward   #'unembedding-fwd
                 :backward  #'unembedding-bwd)
