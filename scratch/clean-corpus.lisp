;;;; scratch/clean-corpus.lisp
;;;;
;;;; Cleans a text corpus by removing characters that don't belong:
;;;;   - NUL bytes (always removed; sign of binary contamination)
;;;;   - Characters appearing fewer than :min-freq times (default 5)
;;;;
;;;; Two-pass: first pass counts character frequencies, second pass
;;;; writes only characters that pass the filter. This lets us drop
;;;; rare junk (OCR artifacts, mojibake byproducts, stray symbols)
;;;; without having to enumerate them by hand.
;;;;
;;;; How to run (from the project root):
;;;;   (load (compile-file "scratch/clean-corpus.lisp"))
;;;;   (clean-corpus "corpus/poeall.txt" "corpus/poeall-clean.txt")
;;;;   (clean-corpus "corpus/poeall.txt" "corpus/poeall-clean.txt" :min-freq 10)

(defun clean-corpus (input-path output-path &key (min-freq 5))
  "Copy INPUT-PATH to OUTPUT-PATH, keeping only characters that appear
   at least MIN-FREQ times. NUL bytes are always dropped regardless.
   Returns an alist reporting what was removed."
  (let ((freq (make-hash-table))
        (text nil))
    ;; --- Pass 1: load file and count character frequencies ---
    (with-open-file (in input-path :direction :input)
      (setf text (make-string (file-length in)))
      (read-sequence text in))
    (loop for c across text do
      (incf (gethash c freq 0)))
    ;; --- Build the keep-set: freq >= min-freq AND not NUL ---
    (let ((keep (make-hash-table))
          (dropped-chars '())
          (dropped-count 0))
      (maphash (lambda (c n)
                 (if (and (>= n min-freq)
                          (not (zerop (char-code c))))
                     (setf (gethash c keep) t)
                     (progn
                       (push (cons c n) dropped-chars)
                       (incf dropped-count n))))
               freq)
      ;; --- Pass 2: write only kept characters ---
      (with-open-file (out output-path :direction :output
                                       :if-exists :supersede
                                       :if-does-not-exist :create)
        (loop for c across text
              when (gethash c keep)
                do (write-char c out)))
      ;; --- Report ---
      (format t "Input:  ~A (~D chars)~%" input-path (length text))
      (format t "Output: ~A~%" output-path)
      (format t "Kept ~D unique chars, dropped ~D unique chars (~D occurrences)~%"
              (hash-table-count keep)
              (length dropped-chars)
              dropped-count)
      (format t "Dropped chars (count, code, char):~%")
      (dolist (pair (sort dropped-chars #'> :key #'cdr))
        (format t "  ~6D  U+~4,'0X  ~S~%"
                (cdr pair) (char-code (car pair)) (car pair)))
      (list (cons :kept-unique (hash-table-count keep))
            (cons :dropped-unique (length dropped-chars))
            (cons :dropped-occurrences dropped-count)))))
