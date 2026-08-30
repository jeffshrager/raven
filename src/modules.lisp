;;;; src/modules.lisp
;;;;
;;;; Modules are the composable units the DSL wires together. Each module
;;;; type has three registered functions:
;;;;
;;;;   allocator : (config) -> params-plist
;;;;   forward   : (params input context) -> (values output saved)
;;;;   backward  : (params saved grad-out) -> (values grad-input grad-params)
;;;;
;;;; Registry: hashtable keyed by module-type keyword, mapping to a plist
;;;; of (:allocator :forward :backward). Adding a new module = one file,
;;;; one register-module call.
;;;;
;;;; Modules in this file:
;;;;   :embedding             token id -> vector
;;;;   :positional-sinusoidal add sinusoidal position encoding
;;;;   :rmsnorm               root-mean-square norm with learned scale
;;;;   :attention-block       multi-head causal self-attention
;;;;   :ffn                   two-layer MLP with GELU
;;;;   :transformer-block     pre-norm: (rmsnorm -> attn -> res) then
;;;;                          (rmsnorm -> ffn -> res)
;;;;   :unembedding           final projection to vocab logits
;;;;
;;;; Shape convention throughout: activations are (T d_model). No batch
;;;; dimension. One sequence at a time.
;;;;
;;;; Weight-matrix convention: W is stored as (d_in, d_out) so
;;;; matmul-forward(input, W) works directly with no transpose. This
;;;; applies to attention (Wq, Wk, Wv, Wo), FFN (w1, w2), and unembedding.
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
  "Install the three functions that make up module type TYPE (a
   keyword like :rmsnorm). Called once per module, at the bottom of
   this file, right after that module's ALLOC/FWD/BWD are defined."
  (setf (gethash type *module-registry*)
        (list :allocator allocator :forward forward :backward backward)))

(defun get-module (type)
  "Look up TYPE's (:allocator :forward :backward) plist, or error if
   no module of that type was ever REGISTER-MODULE'd — this is the
   'unknown module type' guard for typos in a spec (see model.lisp)."
  (or (gethash type *module-registry*)
      (error "Unknown module type: ~S" type)))

(defun module-alloc (type config)
  "Dispatch to TYPE's allocator: CONFIG (a plist of hyperparameters
   like :d-model) in, a fresh params plist out."
  (funcall (getf (get-module type) :allocator) config))

(defun module-forward (type params input context)
  "Dispatch to TYPE's forward function. CONTEXT is currently unused by
   every module in this file (each ignores it) but is threaded through
   uniformly so a future module type — one that needs to know its
   position in the model, e.g. for KV-caching — can use it without
   changing this dispatcher's signature."
  (funcall (getf (get-module type) :forward) params input context))

(defun module-backward (type params saved grad-out)
  "Dispatch to TYPE's backward function: SAVED is exactly whatever
   that type's forward returned as its second value."
  (funcall (getf (get-module type) :backward) params saved grad-out))


;;; ==========================================================================
;;; Small helpers used by multiple modules
;;; ==========================================================================
;;;
;;; ADD-INTO! is used wherever a residual or a multi-branch gradient
;;; contribution has to be summed. SLICE-HEAD / SCATTER-HEAD! move a
;;; (T, dk) slice in and out of a (T, H*dk) full-width tensor; used by
;;; multi-head attention forward AND backward, which used to be inlined
;;; four times each and produced a paren-thicket.

(defun copy-tensor! (dest src)
  "Elementwise copy src into dest. Same shape required."
  (declare (type tensor dest src))
  (dotimes (i (array-total-size dest))
    (setf (row-major-aref dest i) (row-major-aref src i)))
  dest)

(defun add-into! (dest a b)
  "dest[i] = a[i] + b[i]. Same shape required."
  (declare (type tensor dest a b))
  (dotimes (i (array-total-size dest))
    (setf (row-major-aref dest i)
          (+ (row-major-aref a i) (row-major-aref b i))))
  dest)

(defun slice-head (src head-idx dk)
  "Extract head HEAD-IDX from SRC (T, H*dk). Returns fresh (T, dk).
   Columns [head*dk : (head+1)*dk] of SRC become columns [0 : dk] of dest.

   Worked example: D=8, H=4 heads, so dk=D/H=2. SRC's 8 columns are 4
   consecutive 2-wide blocks, one per head:
       columns:  [0 1 | 2 3 | 4 5 | 6 7]
       head:        0     1     2     3
   SLICE-HEAD src 1 2 computes start=(* 1 2)=2 and copies SRC's
   columns 2..3 (inclusive) into a fresh (T 2) tensor's columns 0..1.
   This is the D-dimensional embedding split into H independent
   dk-dimensional subspaces that standard multi-head attention uses —
   each head attends using only its own slice of Q/K/V."
  (declare (type tensor-2d src)
           (type fixnum head-idx dk))
  (let* ((t-len (array-dimension src 0))
         (dest  (make-tensor (list t-len dk)))
         (start (* head-idx dk)))
    (declare (type fixnum t-len start)
             (type tensor-2d dest))
    (dotimes (tt t-len)
      (dotimes (kk dk)
        (setf (aref dest tt kk) (aref src tt (+ start kk)))))
    dest))

(defun scatter-head! (dest head-tensor head-idx dk)
  "Copy HEAD-TENSOR (T, dk) into columns [head*dk : (head+1)*dk] of DEST.
   The exact inverse of SLICE-HEAD (see its worked example) — used to
   reassemble each head's independently-computed (T dk) output back
   into its own column-block of the full (T D) tensor, so the H heads'
   results sit side by side exactly where they were split from."
  (declare (type tensor-2d dest head-tensor)
           (type fixnum head-idx dk))
  (let ((t-len (array-dimension dest 0))
        (start (* head-idx dk)))
    (declare (type fixnum t-len start))
    (dotimes (tt t-len)
      (dotimes (kk dk)
        (setf (aref dest tt (+ start kk)) (aref head-tensor tt kk)))))
  dest)


;;; ==========================================================================
;;; :embedding    token id -> row of learned table
;;; ==========================================================================
;;;
;;; Config: (:vocab-size N :d-model D)
;;; Params: (:table (V D))
;;; Input:  (simple-array fixnum (T))
;;; Output: (T D)

(defun embedding-alloc (config)
  (list :table (xavier-init (getf config :vocab-size) (getf config :d-model))))

(defun embedding-fwd (params input context)
  "INPUT is (simple-array fixnum (T)) of token ids. Output is (T D):
   row t is TABLE's row for token INPUT[t] — a straight lookup, no
   arithmetic. See EMBEDDING-LOOKUP-FORWARD in tensor-ops.lisp for the
   actual index/copy loop."
  (declare (ignore context))
  (embedding-lookup-forward (getf params :table) input))

(defun embedding-bwd (params saved grad-out)
  "GRAD-OUT (T D) scatters back into a (V D) grad for TABLE: row
   INPUT[t]'s gradient accumulates grad-out row t (accumulates, not
   overwrites, because the same token id can appear at multiple
   positions t — every occurrence pushes the embedding in its own
   direction, and those pushes sum). IDs are discrete, not
   differentiable, so there is no gradient w.r.t. INPUT itself — the
   first return value is NIL rather than a tensor."
  (declare (ignore params))
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
;;; Params: (:pe (L D)) -- treated as constant, no gradient
;;; Input/Output: (T D)
;;;
;;; PE[pos, 2i]   = sin(pos / 10000^(2i/D))
;;; PE[pos, 2i+1] = cos(pos / 10000^(2i/D))

(defun build-sinusoidal-pe (max-len d)
  (let ((pe (make-tensor (list max-len d))))
    (declare (type tensor-2d pe))
    (dotimes (pos max-len)
      (dotimes (i d)
        (let* ((two-i (* 2 (floor i 2)))
               (freq  (expt 10000.0f0 (/ (float two-i 1.0f0)
                                         (float d     1.0f0))))
               (angle (/ (float pos 1.0f0) freq)))
          (setf (aref pe pos i)
                (if (evenp i) (sin angle) (cos angle))))))
    pe))

(defun positional-sinusoidal-alloc (config)
  (list :pe (build-sinusoidal-pe (getf config :max-len)
                                 (getf config :d-model))))

(defun positional-sinusoidal-fwd (params input context)
  "Add the fixed (L D) sinusoidal table's first T rows to INPUT (T D)
   elementwise: PE encodes each position's index into the D-dim space
   via sin/cos at geometrically-spaced frequencies (see BUILD-SINUSOIDAL-PE),
   so after this addition the model's activations carry both token
   identity (from the embedding) and position (from here) in the same
   vector, added rather than concatenated to keep D unchanged."
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
  "d(x+PE)/dx = identity, so grad_input is just grad_out copied
   through unchanged. Returns NIL for the params-gradient slot: PE is
   a fixed function of position, not a learned parameter (see the
   :PE-skipping logic in COLLECT-TENSORS, training.lisp), so it never
   receives or needs a gradient of its own."
  (declare (ignore params saved))
  (values (copy-tensor! (tensor-like grad-out) grad-out) nil))

(register-module :positional-sinusoidal
                 :allocator #'positional-sinusoidal-alloc
                 :forward   #'positional-sinusoidal-fwd
                 :backward  #'positional-sinusoidal-bwd)


;;; ==========================================================================
;;; :rmsnorm    y[t,k] = (x[t,k] / rms(x[t,:])) * gamma[k]
;;; ==========================================================================
;;;
;;; Config: (:d-model D :eps 1e-5)
;;; Params: (:gamma (D)) ones-initialized
;;;
;;; Per-row: ms  = mean(x^2) + eps
;;;          rms = sqrt(ms)
;;;          inv = 1/rms
;;;          y[k] = x[k] * inv * gamma[k]
;;;
;;; Backward (per row):
;;;   s = sum_k grad_out[k] * gamma[k] * x[k]
;;;   grad_x[j] = inv * grad_out[j] * gamma[j]  -  x[j] * s / (D * ms * rms)
;;;             = inv * grad_out[j] * gamma[j]  -  x[j] * inv * s / (D * ms)
;;;   grad_gamma[k] += grad_out[k] * x[k] * inv    (summed across rows)

(defun rmsnorm-alloc (config)
  (list :gamma (ones (list (getf config :d-model)))
        :eps   (coerce (or (getf config :eps) 1.0f-5) 'single-float)))

(defun rmsnorm-fwd (params input context)
  "Row-wise RMS normalization: for each row t of INPUT (T D), rescales
   the row to have root-mean-square magnitude 1 (up to EPS for
   numerical safety), then applies a learned per-channel scale GAMMA.
   Unlike LayerNorm, there's no mean-subtraction or learned bias —
   RMSNorm only controls scale, not shift, which is cheaper and works
   just as well in practice for transformer blocks.

   Saves (INPUT INVS MSS) for backward: INVS[t] = 1/rms(row t) and
   MSS[t] = mean-square(row t)+EPS are exactly the per-row
   intermediates the backward math needs (see the header derivation
   above) — recomputing them from INPUT alone in the backward would
   need the same sum-of-squares loop again, so we just keep them."
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
  "Backward for RMSNORM-FWD; see the header derivation above for the
   grad_x / grad_gamma formulas. Each row is independent (RMSNorm
   normalizes per-row, not across rows), so the whole computation is a
   per-row loop; GRAD-GAMMA accumulates across rows with INCF because
   GAMMA is shared by every row (one set of D scale parameters used T
   times, once per position — like the embedding table, every use
   contributes to the same gradient)."
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
;;; Config: (:d-model D :n-heads H)     D must be divisible by H
;;; Params: (:wq (D D) :wk (D D) :wv (D D) :wo (D D) :n-heads H)
;;;         (No biases — modern LLM default.)
;;; Input/Output: (T D)
;;;
;;; Per head h (dk = D/H):
;;;   Slice Qh, Kh, Vh out of full Q=X.Wq, K=X.Wk, V=X.Wv  (each (T, dk))
;;;   S  = Qh . Kh^T * (1/sqrt(dk))               (T, T)
;;;   S[i,j] = -inf for j>i                       causal mask
;;;   A  = softmax(S) row-wise                    (T, T)
;;;   Zh = A . Vh                                 (T, dk)
;;; Scatter Zh back into full Z (T, D).
;;; Y = Z . Wo                                    (T, D)

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
  "Multi-head causal self-attention, forward pass.

   Shape trace (T = sequence length, D = d-model, H = n-heads, dk = D/H):
     INPUT (T D)
       -> Q, K, V = INPUT.Wq, INPUT.Wk, INPUT.Wv     each (T D)  [one matmul
          each over the FULL width — cheaper than H separate small matmuls,
          and equivalent because slicing columns commutes with the matmul:
          slicing head h out of (INPUT . W) equals INPUT . (columns of W
          for head h). So we project once, then slice.]
     per head h in [0, H):
       Qh, Kh, Vh = SLICE-HEAD of Q, K, V at head h     each (T dk)
       KhT = transpose(Kh)                              (dk T)
       S = Qh . KhT                                     (T T)  [S[i,j] is
          the raw dot-product affinity of query position i for key
          position j, before scaling or masking]
       S <- S * (1/sqrt(dk)), with S[i,j] = -inf for j > i   (still (T T))
       A = softmax(S), row-wise                          (T T)  [row i is a
          probability distribution over which earlier positions i attends to]
       Zh = A . Vh                                       (T dk)
     Z = SCATTER-HEAD! of all Zh back into columns        (T D)
     OUTPUT = Z . Wo                                      (T D)

   Causal mask worked example (T=4): row i of S is query position i;
   column j is key position j. Masking j > i zeroes out (via -inf, so
   softmax sends it to 0 probability) every key STRICTLY AFTER the
   query, i.e. row 0 can only attend to column 0; row 3 can attend to
   columns 0..3. This is what makes the model a causal (left-to-right)
   language model: position i's output can never depend on tokens that
   come after it.
       row i=0: [ S00  -inf -inf -inf ]   attends to {0}
       row i=1: [ S10   S11 -inf -inf ]   attends to {0,1}
       row i=2: [ S20   S21  S22 -inf ]   attends to {0,1,2}
       row i=3: [ S30   S31  S32  S33 ]   attends to {0,1,2,3}

   Saves everything BACKWARD needs to redo this per-head slicing and
   invert the causal mask: the per-head softmax outputs (ATTN — needed
   because SOFTMAX-BACKWARD's Jacobian depends on the forward output,
   not the input) plus the projected Q/K/V/Z and the weight matrices."
  (declare (ignore context))
  (let* ((wq (getf params :wq))
         (wk (getf params :wk))
         (wv (getf params :wv))
         (wo (getf params :wo))
         (h  (getf params :n-heads))
         (t-len (array-dimension input 0))
         (d     (array-dimension input 1))
         (dk    (floor d h))
         (scale-factor (/ 1.0f0 (sqrt (float dk 1.0f0))))
         ;; Project full-width Q, K, V once (no transpose needed thanks to
         ;; (d_in, d_out) storage convention). Shape: (T D) . (D D) -> (T D).
         (q (nth-value 0 (matmul-forward input wq)))
         (k (nth-value 0 (matmul-forward input wk)))
         (v (nth-value 0 (matmul-forward input wv)))
         ;; Z accumulates each head's (T dk) output side-by-side back into
         ;; the full (T D) width — see SCATTER-HEAD! below.
         (z (make-tensor (list t-len d)))
         ;; One softmax-output matrix (T T) per head, saved for backward.
         (attn-all (make-array h)))
    (declare (type tensor-2d input wq wk wv wo q k v z)
             (type fixnum t-len d dk h)
             (type single-float scale-factor))
    (dotimes (hi h)
      (let* ((qh  (slice-head q hi dk))              ; (T dk)
             (kh  (slice-head k hi dk))               ; (T dk)
             (vh  (slice-head v hi dk))               ; (T dk)
             (khT (nth-value 0 (transpose-forward kh))) ; (dk T)
             (s   (nth-value 0 (matmul-forward qh khT)))) ; (T dk).(dk T) = (T T)
        (declare (type tensor-2d qh kh vh khT s))
        ;; Scale and causal-mask in a single pass over S — see the T=4
        ;; worked example in the docstring above for why "j > i" is the
        ;; right condition (row i is the query, column j is the key).
        (dotimes (i t-len)
          (dotimes (j t-len)
            (setf (aref s i j)
                  (if (> j i)
                      most-negative-single-float
                      (* scale-factor (aref s i j))))))
        (let* ((a  (nth-value 0 (softmax-forward s))) ; (T T), rows sum to 1
               (zh (nth-value 0 (matmul-forward a vh)))) ; (T T).(T dk) = (T dk)
          (declare (type tensor-2d a zh))
          (setf (aref attn-all hi) a)
          (scatter-head! z zh hi dk))))
    (values (nth-value 0 (matmul-forward z wo))        ; (T D).(D D) = (T D)
            (list :input input :q q :k k :v v :z z
                  :attn attn-all
                  :wq wq :wk wk :wv wv :wo wo
                  :h h :dk dk :scale scale-factor))))

(defun attention-bwd (params saved grad-out)
  "Backward pass for ATTENTION-FWD. Mirrors the forward's shape trace
   in reverse, one MATMUL-BACKWARD / SOFTMAX-BACKWARD per forward op:
     GRAD-OUT (T D) --[Y=Z.Wo]--> grad-z (T D), grad-wo (D D)
     per head: grad-zh (T dk) --[Zh=A.Vh]--> grad-a (T T), grad-vh (T dk)
               grad-a --[A=softmax(S)]--> grad-s (T T), masked/unmasked
                 exactly like the forward's mask (see below)
               grad-s --[S=Qh.KhT]--> grad-qh (T dk), grad-khT (dk T)
               grad-kh = transpose(grad-khT)                    (T dk)
     grad-q, grad-k, grad-v (T D) accumulate the scattered per-head
       grad-qh/grad-kh/grad-vh, then each is pushed back through its
       own projection (Q=input.Wq etc.) to get a grad-input contribution
       and a grad-Wq/Wk/Wv.
     grad-input is the SUM of the three projections' contributions,
       because INPUT is used three times in the forward pass (once each
       to produce Q, K, V) — by the multivariate chain rule, gradients
       from every use of a value add together.

   Masking the gradient: the causal mask in the forward pass sets
   S[i,j] to -inf for j > i, which softmax turns into A[i,j] = 0 —
   those entries contribute nothing to the loss, so their gradient
   must be exactly 0, not just small. We enforce that explicitly here
   (rather than relying on it falling out of softmax's Jacobian, which
   in principle multiplies by A[i,j]=0 but could in theory pick up
   floating-point noise) by zeroing grad-s at every masked position —
   the same j > i condition as the forward mask."
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
         (grad-q (make-tensor (list t-len d)))
         (grad-k (make-tensor (list t-len d)))
         (grad-v (make-tensor (list t-len d))))
    (declare (type tensor-2d input q k v z wq wk wv wo grad-q grad-k grad-v grad-out)
             (type fixnum t-len d dk h)
             (type single-float scale-factor))
    ;; Y = Z . Wo  ->  grad_z, grad_wo
    (multiple-value-bind (grad-z grad-wo)
        (matmul-backward grad-out (list z wo))
      (declare (type tensor-2d grad-z grad-wo))
      ;; Per-head backward.
      (dotimes (hi h)
        (let* ((qh      (slice-head q hi dk))
               (kh      (slice-head k hi dk))
               (vh      (slice-head v hi dk))
               (grad-zh (slice-head grad-z hi dk))
               (a       (aref attn hi))
               (khT     (nth-value 0 (transpose-forward kh))))
          (declare (type tensor-2d qh kh vh grad-zh a khT))
          ;; Zh = A . Vh  ->  grad_a, grad_vh
          (multiple-value-bind (grad-a grad-vh)
              (matmul-backward grad-zh (list a vh))
            (declare (type tensor-2d grad-a grad-vh))
            ;; A = softmax(masked S)  ->  grad_s (before mask/scale)
            (let ((grad-s (softmax-backward grad-a a)))
              (declare (type tensor-2d grad-s))
              ;; Zero masked positions and apply scale factor in one pass.
              (dotimes (i t-len)
                (dotimes (j t-len)
                  (if (> j i)
                      (setf (aref grad-s i j) 0.0f0)
                      (setf (aref grad-s i j)
                            (* scale-factor (aref grad-s i j))))))
              ;; S = Qh . Kh^T  ->  grad_qh, grad_khT
              (multiple-value-bind (grad-qh grad-khT)
                  (matmul-backward grad-s (list qh khT))
                (declare (type tensor-2d grad-qh grad-khT))
                (let ((grad-kh (nth-value 0 (transpose-forward grad-khT))))
                  (declare (type tensor-2d grad-kh))
                  (scatter-head! grad-q grad-qh hi dk)
                  (scatter-head! grad-k grad-kh hi dk)
                  (scatter-head! grad-v grad-vh hi dk)))))))
      ;; Project grad-Q, grad-K, grad-V back through their input projections.
      ;; Forward: Q = input . wq (etc.); no transpose in storage.
      (multiple-value-bind (gi-q grad-wq) (matmul-backward grad-q (list input wq))
        (multiple-value-bind (gi-k grad-wk) (matmul-backward grad-k (list input wk))
          (multiple-value-bind (gi-v grad-wv) (matmul-backward grad-v (list input wv))
            (declare (type tensor-2d gi-q gi-k gi-v grad-wq grad-wk grad-wv))
            (let ((grad-input (tensor-like input)))
              (declare (type tensor-2d grad-input))
              ;; grad_input = gi_q + gi_k + gi_v
              (dotimes (i (array-total-size input))
                (setf (row-major-aref grad-input i)
                      (+ (row-major-aref gi-q i)
                         (row-major-aref gi-k i)
                         (row-major-aref gi-v i))))
              (values grad-input
                      (list :wq grad-wq :wk grad-wk :wv grad-wv :wo grad-wo)))))))))

(register-module :attention-block
                 :allocator #'attention-alloc
                 :forward   #'attention-fwd
                 :backward  #'attention-bwd)


;;; ==========================================================================
;;; :ffn    two-layer MLP with GELU activation
;;; ==========================================================================
;;;
;;; Config: (:d-model D :d-ff F)   F usually 4*D
;;; Params: (:w1 (D F) :b1 (F) :w2 (F D) :b2 (D))
;;;
;;;   h1a = input . w1               (T F)
;;;   h1  = h1a + b1                 (T F)
;;;   h2  = gelu(h1)                 (T F)
;;;   h3  = h2 . w2                  (T D)
;;;   y   = h3 + b2                  (T D)
;;;
;;; Save only what the backward needs: input, h2 (for grad_w2), gelu-saved
;;; (for gelu-backward), and the weight matrices w1, w2 (matmul-backward
;;; needs them). The intermediate activations h1a, h1, h3 are not needed.

(defun ffn-alloc (config)
  (let ((d (getf config :d-model))
        (f (or (getf config :d-ff) (* 4 (getf config :d-model)))))
    (list :w1 (xavier-init d f)      ; (D F) under (in, out) convention
          :b1 (zeros (list f))
          :w2 (xavier-init f d)      ; (F D)
          :b2 (zeros (list d)))))

(defun ffn-fwd (params input context)
  "Position-wise feed-forward network: expand (T D) -> (T F) with F
   usually 4x D, apply a GELU nonlinearity, then contract back to
   (T D). This is where per-position \"thinking\" happens in a
   transformer block — attention only MOVES information between
   positions, it has no nonlinearity of its own; the FFN is what lets
   the model compute a nonlinear function of what attention gathered.
   Every position is processed independently and identically (same W1/W2
   for every row t), which is why this is a plain matmul rather than
   anything attention-like."
  (declare (ignore context))
  (let* ((w1 (getf params :w1))
         (b1 (getf params :b1))
         (w2 (getf params :w2))
         (b2 (getf params :b2))
         (h1a (nth-value 0 (matmul-forward input w1)))
         (h1  (nth-value 0 (add-bias-forward h1a b1))))
    (multiple-value-bind (h2 gelu-saved) (gelu-forward h1)
      (let* ((h3 (nth-value 0 (matmul-forward h2 w2)))
             (y  (nth-value 0 (add-bias-forward h3 b2))))
        (values y (list :input input :h2 h2
                        :w1 w1 :w2 w2 :gelu-saved gelu-saved))))))

(defun ffn-bwd (params saved grad-out)
  "Backward for FFN-FWD — a straight-line chain of 4 ops (add-bias,
   matmul, gelu, add-bias, matmul in forward order), so this is just
   the chain rule applied once per op in reverse, each step consuming
   the previous step's grad and producing the next one plus that op's
   own param gradient(s). No branching or accumulation needed (unlike
   attention or embedding) because every value here is used exactly
   once downstream."
  (declare (ignore params))
  (let ((input (getf saved :input))
        (h2    (getf saved :h2))
        (w1    (getf saved :w1))
        (w2    (getf saved :w2))
        (gs    (getf saved :gelu-saved)))
    ;; y = h3 + b2 -> grad_h3, grad_b2
    (multiple-value-bind (grad-h3 grad-b2) (add-bias-backward grad-out nil)
      ;; h3 = h2 . w2 -> grad_h2, grad_w2
      (multiple-value-bind (grad-h2 grad-w2) (matmul-backward grad-h3 (list h2 w2))
        ;; h2 = gelu(h1) -> grad_h1
        (let ((grad-h1 (gelu-backward grad-h2 gs)))
          ;; h1 = h1a + b1 -> grad_h1a, grad_b1
          (multiple-value-bind (grad-h1a grad-b1) (add-bias-backward grad-h1 nil)
            ;; h1a = input . w1 -> grad_input, grad_w1
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
;;; Params: (:norm1 <rmsnorm params> :attn <attn params>
;;;          :norm2 <rmsnorm params> :ffn  <ffn params>)
;;;
;;; Forward (pre-norm):
;;;   a  = attention(rmsnorm(x))
;;;   x1 = x + a               residual
;;;   f  = ffn(rmsnorm(x1))
;;;   y  = x1 + f              residual
;;;
;;; Backward: reverse composition; residuals mean grad flows into both
;;; branches of each add — accumulate with ADD-INTO!.

(defun transformer-block-alloc (config)
  (list :norm1 (module-alloc :rmsnorm         config)
        :attn  (module-alloc :attention-block config)
        :norm2 (module-alloc :rmsnorm         config)
        :ffn   (module-alloc :ffn             config)))

(defun transformer-block-fwd (params input context)
  "One transformer block, PRE-norm style: RMSNorm is applied to the
   INPUT of each sub-layer (attention, then FFN), not its output, and
   each sub-layer's output is added back to the pre-norm residual
   stream (input/x1), not to the normalized value. Pre-norm is the
   standard modern choice over the original post-norm transformer
   because it keeps the residual stream's scale stable across many
   stacked blocks — gradients have a direct, un-normalized path all
   the way through via the two residual adds, which is what makes deep
   stacks of these blocks trainable without careful learning-rate
   warmup tricks."
  (multiple-value-bind (n1 sn1) (module-forward :rmsnorm (getf params :norm1) input context)
    (multiple-value-bind (a sa)  (module-forward :attention-block (getf params :attn) n1 context)
      (let ((x1 (nth-value 0 (add-forward input a))))
        (multiple-value-bind (n2 sn2) (module-forward :rmsnorm (getf params :norm2) x1 context)
          (multiple-value-bind (f sf) (module-forward :ffn (getf params :ffn) n2 context)
            (let ((y (nth-value 0 (add-forward x1 f))))
              (values y (list :sn1 sn1 :sa sa :sn2 sn2 :sf sf)))))))))

(defun transformer-block-bwd (params saved grad-out)
  "Backward for TRANSFORMER-BLOCK-FWD, reversing the two
   residual-add + sub-layer pairs. At each residual add (y = branch +
   skip), ADD-BACKWARD hands the SAME upstream gradient to both the
   branch and the skip connection unchanged (that's just d(a+b)/da =
   d(a+b)/db = 1) — but the skip connection also has its own
   contribution flowing back through the sub-layer above it, so the
   two contributions to grad_x1 (and to grad_input) must be SUMMED
   (via ADD-INTO!), not just taken from one path. This is the concrete
   mechanism behind \"residual connections give gradients a shortcut\":
   grad_input's residual contribution (GRAD-INPUT-A) never passes
   through the attention block or its norm at all."
  (let ((sn1 (getf saved :sn1))
        (sa  (getf saved :sa))
        (sn2 (getf saved :sn2))
        (sf  (getf saved :sf)))
    ;; y = x1 + f  ->  grad_x1_a, grad_f
    (multiple-value-bind (grad-x1-a grad-f) (add-backward grad-out nil)
      ;; f = ffn(n2) -> grad_n2, grad_ffn
      (multiple-value-bind (grad-n2 grad-ffn) (module-backward :ffn (getf params :ffn) sf grad-f)
        ;; n2 = rmsnorm(x1) -> grad_x1_b, grad_norm2
        (multiple-value-bind (grad-x1-b grad-norm2) (module-backward :rmsnorm (getf params :norm2) sn2 grad-n2)
          ;; grad_x1 = grad_x1_a + grad_x1_b
          (let ((grad-x1 (add-into! (tensor-like grad-x1-a) grad-x1-a grad-x1-b)))
            ;; x1 = input + a  ->  grad_input_a, grad_a
            (multiple-value-bind (grad-input-a grad-a) (add-backward grad-x1 nil)
              ;; a = attention(n1) -> grad_n1, grad_attn
              (multiple-value-bind (grad-n1 grad-attn) (module-backward :attention-block (getf params :attn) sa grad-a)
                ;; n1 = rmsnorm(input) -> grad_input_b, grad_norm1
                (multiple-value-bind (grad-input-b grad-norm1) (module-backward :rmsnorm (getf params :norm1) sn1 grad-n1)
                  (values (add-into! (tensor-like grad-input-a) grad-input-a grad-input-b)
                          (list :norm1 grad-norm1
                                :attn  grad-attn
                                :norm2 grad-norm2
                                :ffn   grad-ffn)))))))))))

(register-module :transformer-block
                 :allocator #'transformer-block-alloc
                 :forward   #'transformer-block-fwd
                 :backward  #'transformer-block-bwd)


;;; ==========================================================================
;;; :unembedding    project (T D) -> (T V) logits
;;; ==========================================================================
;;;
;;; Config: (:d-model D :vocab-size V)
;;; Params: (:w (D V))
;;;
;;; No softmax here — loss layer handles it fused for numerical stability.

(defun unembedding-alloc (config)
  (list :w (xavier-init (getf config :d-model) (getf config :vocab-size))))

(defun unembedding-fwd (params input context)
  "Project (T D) hidden states to (T V) raw logits over the vocabulary
   — one score per token per possible next-token. A pure linear map,
   no bias and no softmax: softmax is deferred to the fused
   softmax-cross-entropy loss (SOFTMAX-CE-FWD in training.lisp) for
   numerical stability, and to sampling/generation code at inference
   time — this module's job ends at raw logits."
  (declare (ignore context))
  (matmul-forward input (getf params :w)))

(defun unembedding-bwd (params saved grad-out)
  "Straight MATMUL-BACKWARD — this module IS a matmul, so there's no
   extra structure to unwind."
  (declare (ignore params))
  (multiple-value-bind (grad-input grad-w) (matmul-backward grad-out saved)
    (values grad-input (list :w grad-w))))

(register-module :unembedding
                 :allocator #'unembedding-alloc
                 :forward   #'unembedding-fwd
                 :backward  #'unembedding-bwd)
