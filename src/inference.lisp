;;;; inference.lisp
;;;;
;;;; Text generation for Raven-LLM.
;;;;
;;;; Provides token-level samplers (greedy, temperature multinomial,
;;;; top-k, top-p / nucleus) and a GENERATE function that slides the
;;;; context window to produce arbitrary-length outputs from a trained
;;;; model.
;;;;
;;;; How to run (from repo root in an SBCL REPL):
;;;;   (load (compile-file "src/utilities.lisp"))
;;;;   (load (compile-file "src/tensor-ops.lisp"))
;;;;   (load (compile-file "src/modules.lisp"))
;;;;   (load (compile-file "src/training.lisp"))
;;;;   (load (compile-file "src/model.lisp"))
;;;;   (load (compile-file "src/data.lisp"))
;;;;   (load (compile-file "src/inference.lisp"))
;;;;   (test-generate)   ; trains a tiny model then samples from it
;;;;
;;;; After a real training run:
;;;;   (multiple-value-bind (params spec vocab)
;;;;       (train :corpus-path "corpus/poeall.txt" :steps 5000)
;;;;     (format t "~A~%" (generate params spec vocab "The raven "
;;;;                                :n-tokens 200 :temperature 0.8 :top-k 40)))
;;;;
;;;; Exposes:
;;;;   sample-token        one token from a logits vector
;;;;   generate            variable-length text generation with sliding window
;;;;   test-generate       self-contained sanity test

(in-package :cl-user)


;;; ------------------------------------------------------------------
;;; Single-token samplers
;;; ------------------------------------------------------------------
;;;
;;; All operate on a length-V single-float vector of logits.
;;;
;;; Order of application inside SAMPLE-TOKEN:
;;;   1. If :GREEDY T           → return argmax; ignore everything else.
;;;   2. Scale logits by 1/temperature (temperature < 1 sharpens, > 1
;;;      flattens; = 1 is the model's natural distribution).
;;;   3. If :TOP-K K            → keep only the K largest logits.
;;;   4. If :TOP-P P            → nucleus: keep smallest set whose
;;;                                cumulative softmax prob >= P.
;;;   5. Softmax the remaining logits, sample from the multinomial.
;;;
;;; Convention (matches nanoGPT et al.): temperature is applied BEFORE
;;; filtering. If you want temperature-free filtering, pass :TEMPERATURE
;;; 1.0.

(defun argmax-vec (v)
  "Index of the max element in V."
  (declare (type (simple-array single-float (*)) v))
  (let ((best-i 0)
        (best-x most-negative-single-float))
    (declare (type fixnum best-i)
             (type single-float best-x))
    (dotimes (i (length v))
      (let ((x (aref v i)))
        (declare (type single-float x))
        (when (> x best-x)
          (setf best-x x
                best-i i))))
    best-i))

(defun softmax-vec! (v)
  "In-place stable softmax of V (single-float vector)."
  (declare (type (simple-array single-float (*)) v))
  (let ((mx most-negative-single-float))
    (declare (type single-float mx))
    (dotimes (i (length v))
      (let ((x (aref v i)))
        (when (> x mx) (setf mx x))))
    (let ((sum 0.0f0))
      (declare (type single-float sum))
      (dotimes (i (length v))
        (setf (aref v i) (exp (- (aref v i) mx)))
        (incf sum (aref v i)))
      (dotimes (i (length v))
        (setf (aref v i) (/ (aref v i) sum)))))
  v)

(defun sample-from-probs (probs)
  "Draw one index from a distribution (single-float vector summing to ~1)."
  (declare (type (simple-array single-float (*)) probs))
  (let ((u (random 1.0f0))
        (acc 0.0f0))
    (declare (type single-float u acc))
    (dotimes (i (length probs))
      (incf acc (aref probs i))
      (when (>= acc u) (return-from sample-from-probs i)))
    ;; Numerical drift fallback (sum can end slightly below 1.0)
    (1- (length probs))))

(defun apply-top-k! (logits k)
  "In-place mask: everything outside the top K logits becomes -inf."
  (declare (type (simple-array single-float (*)) logits)
           (type fixnum k))
  (let ((n (length logits)))
    (when (< k n)
      (let* ((sorted    (sort (copy-seq logits) #'>))
             (threshold (aref sorted (1- k))))
        (declare (type single-float threshold))
        (dotimes (i n)
          (when (< (aref logits i) threshold)
            (setf (aref logits i) most-negative-single-float))))))
  logits)

(defun apply-top-p! (logits p)
  "In-place nucleus mask: keep the smallest set of logits whose cumulative
   softmax probability is >= P; set the rest to -inf. Always keeps at
   least one token (the highest-prob one).

   Worked example: probs (after softmax, already sorted descending for
   this illustration) = [0.5, 0.3, 0.15, 0.05], P=0.8. Walking ORDER
   (highest-probability index first) and accumulating:
       after token 1: acc=0.50  (< 0.8, keep going)
       after token 2: acc=0.80  (>= 0.8, STOP — this token is included)
       tokens 3, 4                        : never marked in KEEP, masked
   So the kept set is the top 2 tokens (covering exactly the threshold),
   not the top-k by count — a peaked distribution keeps few tokens, a
   flat one keeps many, which is the whole point of nucleus sampling
   over a fixed top-k."
  (declare (type (simple-array single-float (*)) logits)
           (type single-float p))
  (let* ((n       (length logits))
         (probs   (softmax-vec! (copy-seq logits)))
         (order   (sort (loop for i below n collect i)
                        (lambda (a b) (> (aref probs a) (aref probs b)))))
         (keep    (make-array n :element-type 'bit :initial-element 0))
         (acc     0.0f0))
    (declare (type single-float acc))
    (dolist (i order)
      (setf (sbit keep i) 1)
      (incf acc (aref probs i))
      (when (>= acc p) (return)))
    (dotimes (i n)
      (when (zerop (sbit keep i))
        (setf (aref logits i) most-negative-single-float))))
  logits)

(defun sample-token (logits &key (temperature 1.0f0)
                                 (top-k nil) (top-p nil)
                                 (greedy nil))
  "Sample one token id from LOGITS (single-float vector of length V).
   See file header for the order of operations."
  (declare (type (simple-array single-float (*)) logits)
           (type single-float temperature))
  (when greedy
    (return-from sample-token (argmax-vec logits)))
  (assert (> temperature 0.0f0) () "sample-token: temperature must be > 0")
  ;; Copy + scale — never mutate the caller's logits.
  (let ((scaled (make-array (length logits) :element-type 'single-float)))
    (dotimes (i (length logits))
      (setf (aref scaled i) (/ (aref logits i) temperature)))
    (when top-k (apply-top-k! scaled top-k))
    (when top-p (apply-top-p! scaled top-p))
    (sample-from-probs (softmax-vec! scaled))))


;;; ------------------------------------------------------------------
;;; Sliding-window generation
;;; ------------------------------------------------------------------
;;;
;;; The trained model has a fixed max context T. To generate more than
;;; T tokens, at each step we take the last T (or fewer, if we don't
;;; have T yet) tokens as input, forward, read the LAST row of logits
;;; (position T-1, the "next token" prediction), sample, append.
;;;
;;; Positional-sinusoidal is safe with variable T ≤ max-len — it only
;;; touches PE rows 0..T-1. Attention likewise operates on the actual
;;; T of the input, not the training-time T.

(defun infer-max-context (spec)
  "Look up the model's max context by finding the first module in SPEC
   with a :MAX-LEN in its config. Returns NIL if none found."
  (loop for (name type config) in spec
        for m = (getf config :max-len)
        when m return m))

(defun generate (params spec vocab prompt
                 &key (n-tokens 100)
                      (temperature 1.0f0)
                      (top-k nil) (top-p nil)
                      (greedy nil)
                      (max-t nil))
  "Generate N-TOKENS characters after PROMPT and return the full string
   (prompt + generated). PARAMS/SPEC/VOCAB come from TRAIN.
   MAX-T defaults to the model's :MAX-LEN (from POSITIONAL-SINUSOIDAL).

   Each iteration: LOGITS from a forward pass over the current WINDOW
   (length <= M-T) has shape (T_window V) — row t is the model's
   next-token prediction made using only tokens 0..t of the window (a
   consequence of the causal mask in ATTENTION-FWD). So row T_window-1
   is the prediction for the token that comes after EVERY token
   currently in the window — exactly the next character we want to
   sample. That's why only the last row is read out (LAST, below),
   even though the forward pass computed a full (T_window V) matrix
   of predictions for every position."
  (let ((fwd   (make-model-fwd spec))
        (m-t   (or max-t (infer-max-context spec))))
    (assert m-t () "GENERATE: could not determine max context; pass :MAX-T explicitly.")
    (let* ((prompt-ids (encode vocab prompt))
           ;; Use a simple growing list; small overhead, easy to reason about.
           (all-ids (coerce prompt-ids 'list)))
      (dotimes (_ n-tokens)
        (let* ((len    (length all-ids))
               (start  (max 0 (- len m-t)))
               (window (subseq all-ids start))
               (input  (make-array (length window)
                                   :element-type 'fixnum
                                   :initial-contents window))
               (logits (nth-value 0 (funcall fwd params input)))
               (t-len  (array-dimension logits 0))
               (v      (array-dimension logits 1))
               (last   (make-array v :element-type 'single-float)))
          ;; Copy last row of logits (the next-token prediction).
          (dotimes (j v)
            (setf (aref last j) (aref logits (1- t-len) j)))
          (let ((next (sample-token last :temperature temperature
                                         :top-k top-k :top-p top-p
                                         :greedy greedy)))
            (setf all-ids (nconc all-ids (list next))))))
      (decode vocab (make-array (length all-ids)
                                :element-type 'fixnum
                                :initial-contents all-ids)))))


;;; ------------------------------------------------------------------
;;; Self-contained test: train briefly, then sample
;;; ------------------------------------------------------------------
;;;
;;; If everything is wired correctly, a model overtrained on a repeating
;;; phrase should generate that phrase (or a close cousin) when sampled.
;;; We do a short training run, then sample greedily and with a low
;;; temperature to reduce noise, and print both.

(defun test-generate (&key (steps 400) (n-tokens 60))
  (format t "~&=== test-generate: overfit a repeating phrase, then sample ===~%")
  (let* ((text   (with-output-to-string (s)
                   (dotimes (_ 200) (write-string "the quick brown fox " s))))
         (vocab  (build-vocab text))
         (v      (vocab-size vocab))
         (corpus (encode vocab text))
         (cfg    (list :vocab-size v :d-model 32 :n-heads 4
                       :n-layers 1 :max-len 16 :d-ff 64 :eps 1.0f-5))
         (spec   (build-gpt-lm-spec cfg))
         (params (build-model-params spec))
         (state  (adam-alloc params))
         (fwd    (make-model-fwd spec))
         (bwd    (make-model-bwd spec))
         (loss-sum 0.0f0)
         (loss-n   0))
    (format t "  Training ~D steps (V=~D, T=16)...~%" steps v)
    (dotimes (step steps)
      (multiple-value-bind (input target) (sample-window corpus 16)
        (let ((loss (train-step! fwd bwd params state input target :lr 5.0f-3)))
          (incf loss-sum loss)
          (incf loss-n)
          (when (zerop (mod (1+ step) 100))
            (format t "    step ~4D  avg-loss ~,4E~%"
                    (1+ step) (/ loss-sum loss-n))
            (setf loss-sum 0.0f0 loss-n 0)))))
    (let ((prompt "the quick "))
      (format t "~%  prompt: ~S~%" prompt)
      (format t "  greedy:      ~S~%"
              (generate params spec vocab prompt
                        :n-tokens n-tokens :greedy t))
      (format t "  temp 0.4:    ~S~%"
              (generate params spec vocab prompt
                        :n-tokens n-tokens :temperature 0.4f0))
      (format t "  top-k=3:     ~S~%"
              (generate params spec vocab prompt
                        :n-tokens n-tokens :top-k 3 :temperature 1.0f0))
      (format t "  top-p=0.9:   ~S~%"
              (generate params spec vocab prompt
                        :n-tokens n-tokens :top-p 0.9f0 :temperature 1.0f0)))
    (format t "~%  (For a corpus this trivial, all sampling modes should~%")
    (format t "   reproduce the periodic phrase — small deviations OK.)~%")))
