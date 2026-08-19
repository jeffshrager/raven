;;;; src/tokenizer.lisp
;;;;
;;;; Character-level tokenizer.
;;;;
;;;; Vocabulary is built from a training corpus: every unique character
;;;; in the corpus becomes a token, sorted by Unicode code point so the
;;;; assignment is reproducible across runs. There are no special tokens
;;;; (<BOS>, <EOS>, <PAD>, <UNK>) — for a single-corpus training run
;;;; the vocab is a closed set by construction.
;;;;
;;;; The vocab is stored as a defstruct with three fields:
;;;;   - size       : integer, number of tokens
;;;;   - id->char   : simple-vector, index = id, value = character
;;;;   - char->id   : hashtable, key = character, value = id
;;;; Two lookup structures instead of one because encode wants fast
;;;; char->id (hashtable) and decode wants fast id->char (vector aref).
;;;;
;;;; Persistence: the vocab file is a Lisp alist of (id . char) pairs
;;;; written with PRINT, read back with READ. Human-readable, no custom
;;;; serialization code.
;;;;
;;;; How to run (from the project root):
;;;;   (load (compile-file "src/tokenizer.lisp"))
;;;;   ;; Build vocab from a corpus string:
;;;;   (defparameter *v* (build-vocab (load-corpus "corpus/poeall.txt")))
;;;;   ;; Encode and decode:
;;;;   (defparameter *ids* (encode "Once upon a midnight" *v*))
;;;;   (decode *ids* *v*)
;;;;   ;; Save/load:
;;;;   (save-vocab *v* "results/vocab.lisp")
;;;;   (defparameter *v2* (load-vocab "results/vocab.lisp"))
;;;;
;;;; NOTE: LOAD-CORPUS is defined in src/utilities.lisp (once we factor
;;;; it out) or duplicated inline in the calling script for now.

#|

(load (compile-file "src/tokenizer.lisp"))
;; assuming load-corpus is available:
(defparameter *v* (build-vocab (load-corpus "corpus/poeall.txt")))
(vocab-size *v*)                         ; should be ~110
(decode (encode "Nevermore." *v*) *v*)   ; should return "Nevermore."
(save-vocab *v* "results/vocab.lisp")
(defparameter *v2* (load-vocab "results/vocab.lisp"))
(= (vocab-size *v*) (vocab-size *v2*))   ; T
(decode (encode "quoth the raven" *v2*) *v2*)  ; round-trip via loaded vocab

|#

(defun load-corpus (path)
  "Read entire file at PATH into a single string. Uses read-sequence's
   return value to handle the case where char count < byte count
   (multi-byte UTF-8 makes file-length an overestimate of char count)."
  (with-open-file (stream path :direction :input)
    (let* ((buffer (make-string (file-length stream)))
           (actual (read-sequence buffer stream)))
      (subseq buffer 0 actual))))


;;; ---------- Vocab structure ----------
;;; Defstruct gives us a constructor (MAKE-VOCAB), predicate (VOCAB-P),
;;; and accessors (VOCAB-SIZE, VOCAB-ID->CHAR, VOCAB-CHAR->ID) for free.

(defstruct vocab
  (size     0   :type fixnum)
  (id->char #() :type simple-vector)
  (char->id (make-hash-table) :type hash-table))

;;; ---------- Building the vocab ----------
;;; Scan the corpus, collect the unique character set, sort by char code
;;; for reproducibility, then populate both lookup structures.

(defun build-vocab (text)
  "Build a vocab from TEXT (a string). Every unique character becomes a
   token; ids are assigned in order of ascending char code."
  (let ((seen (make-hash-table)))
    (loop for c across text do
      (setf (gethash c seen) t))
    ;; Extract unique chars, sort by code point.
    (let* ((chars (let ((acc '()))
                    (maphash (lambda (k v) (declare (ignore v)) (push k acc))
                             seen)
                    (sort acc #'< :key #'char-code)))
           (size (length chars))
           (i->c (make-array size))
           (c->i (make-hash-table)))
      (loop for c in chars
            for i from 0
            do (setf (aref i->c i) c
                     (gethash c c->i) i))
      (make-vocab :size size :id->char i->c :char->id c->i))))

;;; ---------- Encode / decode ----------
;;; Encode returns (SIMPLE-ARRAY FIXNUM (*)) so the training loop can
;;; index and slice it without cons overhead. Decode returns a string.
;;;
;;; ENCODE will error via GETHASH returning NIL if TEXT contains a char
;;; not in the vocab. That's intentional: with no <UNK> token, an
;;; out-of-vocab char is a bug (probably a corpus mismatch) and should
;;; be caught loudly, not silently mapped.

(defun encode (text vocab)
  "Convert TEXT (string) to a (SIMPLE-ARRAY FIXNUM (*)) of token ids."
  (let* ((n (length text))
         (out (make-array n :element-type 'fixnum))
         (table (vocab-char->id vocab)))
    (loop for i from 0 below n
          for c = (char text i)
          for id = (gethash c table)
          do (unless id
               (error "Character ~S (U+~4,'0X) not in vocab."
                      c (char-code c)))
             (setf (aref out i) id))
    out))

(defun decode (ids vocab)
  "Convert IDS (vector of fixnums) to a string."
  (let* ((n (length ids))
         (out (make-string n))
         (table (vocab-id->char vocab)))
    (loop for i from 0 below n
          do (setf (char out i) (aref table (aref ids i))))
    out))

;;; ---------- Save / load ----------
;;; The on-disk format is a Lisp alist: ((0 . #\Newline) (1 . #\Space) ...)
;;; written with PRIN1 (readable form, so #\Newline round-trips correctly),
;;; then wrapped in a top-level list with a version tag for future-proofing.

(defun save-vocab (vocab path)
  "Write VOCAB to PATH as a readable Lisp form."
  (let ((alist (loop for i from 0 below (vocab-size vocab)
                     collect (cons i (aref (vocab-id->char vocab) i)))))
    (with-open-file (out path :direction :output
                              :if-exists :supersede
                              :if-does-not-exist :create)
      (let ((*print-readably* t))     ; ensures characters print as #\X
        (prin1 (list :vocab-version 1 :entries alist) out))
      (terpri out)))
  path)

(defun load-vocab (path)
  "Read a vocab file written by SAVE-VOCAB. Rebuilds both lookup tables."
  (with-open-file (in path :direction :input)
    (let* ((form (read in))
           (entries (getf form :entries))
           (size (length entries))
           (i->c (make-array size))
           (c->i (make-hash-table)))
      (dolist (pair entries)
        (let ((id (car pair))
              (c  (cdr pair)))
          (setf (aref i->c id) c
                (gethash c c->i) id)))
      (make-vocab :size size :id->char i->c :char->id c->i))))
