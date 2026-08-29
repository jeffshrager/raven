;;;; data.lisp
;;;;
;;;; Char-level tokenizer and training-window sampler for Raven-LLM.
;;;;
;;;; The corpus (Poe's complete works or similar) is a string. We build
;;;; a vocab of its unique characters and encode the whole corpus once
;;;; to a fixnum array. Training samples random length-(T+1) windows and
;;;; splits them into (input=first T, target=last T = input shifted by 1).
;;;;
;;;; How to run (from repo root in an SBCL REPL):
;;;;   (load (compile-file "src/utilities.lisp"))
;;;;   (load (compile-file "src/data.lisp"))
;;;;   (run-data-tests)   ; vocab roundtrip + sampler sanity
;;;;
;;;; This file is a library, not a script — no top-level side effects at load.
;;;;
;;;; Exposes:
;;;;   VOCAB struct + build-vocab / vocab-size / encode / decode
;;;;   sample-window                — one random (input, target) pair
;;;;   run-data-tests               — sanity checks

(in-package :cl-user)


;;; ------------------------------------------------------------------
;;; Vocabulary
;;; ------------------------------------------------------------------
;;;
;;; We enumerate unique chars in the corpus and assign each a fixnum id
;;; in [0, V). The id-to-char array is sorted by character code so runs
;;; are reproducible regardless of the order chars first appear.
;;;
;;; Encoding produces (SIMPLE-ARRAY FIXNUM (*)) — the type the embedding
;;; module (and the LM training loop) expects for input token ids.

(defstruct vocab
  chars       ; (simple-array character (V)) — id → char
  char->id)   ; hash-table char → fixnum

(defun build-vocab (text)
  "Return a VOCAB containing every unique character in TEXT. Ids are
   assigned by sorted char-code so builds are deterministic across runs
   with the same corpus."
  (declare (type string text))
  (let ((seen (make-hash-table :test 'eql)))
    (loop for c across text do (setf (gethash c seen) t))
    (let* ((unique-chars (sort (loop for c being the hash-keys of seen collect c)
                               #'char<))
           (v            (length unique-chars))
           (chars        (make-array v :element-type 'character
                                       :initial-contents unique-chars))
           (map          (make-hash-table :test 'eql :size v)))
      (dotimes (i v)
        (setf (gethash (aref chars i) map) i))
      (make-vocab :chars chars :char->id map))))

(defun vocab-size (vocab)
  "Number of distinct tokens (character ids) in VOCAB."
  (length (vocab-chars vocab)))

(defun encode (vocab text)
  "Encode TEXT to (SIMPLE-ARRAY FIXNUM (LEN(text))) of token ids.
   Signals an error if TEXT contains a character not in VOCAB."
  (declare (type string text))
  (let* ((n     (length text))
         (ids   (make-array n :element-type 'fixnum))
         (table (vocab-char->id vocab)))
    (dotimes (i n)
      (let ((id (gethash (char text i) table)))
        (unless id
          (error "encode: character ~S at position ~D not in vocab"
                 (char text i) i))
        (setf (aref ids i) id)))
    ids))

(defun decode (vocab ids)
  "Decode a (SIMPLE-ARRAY FIXNUM (*)) of token ids back to a string."
  (declare (type (simple-array fixnum (*)) ids))
  (let* ((n     (length ids))
         (chars (vocab-chars vocab))
         (out   (make-string n)))
    (dotimes (i n)
      (setf (char out i) (aref chars (aref ids i))))
    out))


;;; ------------------------------------------------------------------
;;; Training window sampler
;;; ------------------------------------------------------------------
;;;
;;; Given the encoded corpus and a context length T, pick a random
;;; starting position p in [0, len(corpus) - T - 1] and return
;;;   input  = corpus[p       .. p + T)
;;;   target = corpus[p + 1   .. p + T + 1)
;;; i.e. target is input shifted by one. This is the standard next-token
;;; LM setup: at every position t, the model should predict target[t]
;;; given input[0..t]. Both arrays are freshly allocated so the caller
;;; can hold onto them without worrying about the shared corpus buffer.

(defun sample-window (encoded-corpus context-len)
  "Return (values INPUT TARGET) — two fresh (SIMPLE-ARRAY FIXNUM
   (CONTEXT-LEN)) drawn from a random window of ENCODED-CORPUS.
   TARGET is INPUT shifted by 1 (next-token prediction targets)."
  (declare (type (simple-array fixnum (*)) encoded-corpus)
           (type fixnum context-len))
  (let ((n (length encoded-corpus)))
    (assert (>= n (1+ context-len)) ()
            "sample-window: corpus length ~D too short for context ~D"
            n context-len)
    (let* ((p      (random (- n context-len)))
           (input  (make-array context-len :element-type 'fixnum))
           (target (make-array context-len :element-type 'fixnum)))
      (dotimes (i context-len)
        (setf (aref input  i) (aref encoded-corpus (+ p i)))
        (setf (aref target i) (aref encoded-corpus (+ p i 1))))
      (values input target))))


;;; ------------------------------------------------------------------
;;; Tests
;;; ------------------------------------------------------------------

(defun test-vocab-roundtrip ()
  "Encode → decode of a mixed-case, punctuated string should reproduce
   the original exactly."
  (format t "~&Test: vocab encode/decode roundtrip...~%")
  (let* ((text  "The quick brown fox jumps over the lazy dog. 42!")
         (vocab (build-vocab text))
         (ids   (encode vocab text))
         (back  (decode vocab ids))
         (pass  (string= back text)))
    (format t "  V=~D  match=~A  ~A~%"
            (vocab-size vocab) pass (if pass "PASS" "FAIL"))
    pass))

(defun test-sample-window ()
  "For a known corpus, sampling should satisfy target[i] = input[i+1]
   within the window (target is input shifted). Try 200 samples."
  (format t "~&Test: sample-window shift invariant...~%")
  (let* ((text    "abcdefghijklmnopqrstuvwxyz0123456789")
         (vocab   (build-vocab text))
         (corpus  (encode vocab text))
         (ctx     8)
         (fails   0))
    (dotimes (_ 200)
      (multiple-value-bind (input target) (sample-window corpus ctx)
        (unless (and (= (length input) ctx)
                     (= (length target) ctx)
                     (loop for i below (1- ctx)
                           always (= (aref target i) (aref input (1+ i)))))
          (incf fails))))
    (format t "  200 samples, ~D failures  ~A~%"
            fails (if (zerop fails) "PASS" "FAIL"))
    (zerop fails)))

(defun run-data-tests ()
  "Run all data-module sanity checks. Returns T iff all PASS."
  (format t "~&=== Data module tests ===~%")
  (let ((r1 (test-vocab-roundtrip))
        (r2 (test-sample-window)))
    (format t "~&=== Done ===~%")
    (and r1 r2)))
