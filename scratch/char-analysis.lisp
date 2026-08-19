;;;; scratch/char-analysis.lisp
;;; (load (compile-file "scratch/char-analysis.lisp"))
;;;;
;;;; Character-level analysis of a text corpus. Reports total/unique
;;;; char counts, full frequency table (small enough to print in full),
;;;; and a category breakdown (letters/digits/whitespace/punctuation/other)
;;;; to flag any unusual or non-ASCII characters that might warrant
;;;; normalization before tokenizer construction.
;;;;
;;;; How to run (from the project root):
;;;;   (load (compile-file "scratch/char-analysis.lisp"))
;;;; The (analyze-chars ...) call at the bottom runs automatically.

;;; ---------- File loading ----------
;;; Same as token-analysis; duplicated here so this script is self-contained.

(defun load-corpus (path)
  (with-open-file (stream path :direction :input)
    (let ((contents (make-string (file-length stream))))
      (read-sequence contents stream)
      contents)))

;;; ---------- Counting ----------
;;; Build a frequency table of characters, then sort by count descending.

(defun char-frequencies (text)
  (let ((table (make-hash-table)))       ; eql is fine for characters
    (loop for c across text do
      (incf (gethash c table 0)))
    (let ((pairs '()))
      (maphash (lambda (k v) (push (cons k v) pairs)) table)
      (sort pairs #'> :key #'cdr))))

;;; ---------- Categorization ----------
;;; Bucket each character so we can quickly see if there's anything
;;; the corpus contains that we didn't expect (control chars, exotic
;;; Unicode, etc.). "Other" is the bucket to inspect closely.

(defun categorize (c)
  (let ((code (char-code c)))
    (cond
      ((alpha-char-p c)
       (if (upper-case-p c) :upper :lower))
      ((digit-char-p c)      :digit)
      ((member c '(#\Space #\Tab #\Newline #\Return)) :whitespace)
      ((and (>= code 32) (<= code 126)) :ascii-punct)
      ((< code 32)           :control)        ; non-printing ASCII
      (t                     :other))))       ; non-ASCII (Unicode etc.)

(defun category-summary (freq-alist)
  "Total occurrences and unique-char count per category."
  (let ((totals (make-hash-table))
        (uniques (make-hash-table)))
    (dolist (pair freq-alist)
      (let ((cat (categorize (car pair))))
        (incf (gethash cat totals 0) (cdr pair))
        (incf (gethash cat uniques 0))))
    (let ((rows '()))
      (maphash (lambda (cat count)
                 (push (list cat count (gethash cat uniques)) rows))
               totals)
      (sort rows #'> :key #'second))))

;;; ---------- Reporting ----------

(defun printable-repr (c)
  "Render a character so control chars and whitespace are visible in output."
  (case c
    (#\Newline "\\n")
    (#\Return  "\\r")
    (#\Tab     "\\t")
    (#\Space   "SPC")
    (t (if (< (char-code c) 32)
           (format nil "\\x~2,'0X" (char-code c))
           (string c)))))

(defun print-summary (text freq-alist)
  (let ((total (length text))
        (unique (length freq-alist)))
    (format t "Total characters: ~D~%" total)
    (format t "Unique characters: ~D~%~%" unique)
    (format t "=== Category breakdown ===~%")
    (format t "~12A ~10A ~8A~%" "CATEGORY" "OCCURS" "UNIQUE")
    (dolist (row (category-summary freq-alist))
      (format t "~12A ~10D ~8D~%" (first row) (second row) (third row)))
    (format t "~%=== Full frequency table ===~%")
    (format t "~6A  ~8A  ~A~%" "COUNT" "CODE" "CHAR")
    (dolist (pair freq-alist)
      (format t "~6D  U+~4,'0X  ~A~%"
              (cdr pair)
              (char-code (car pair))
              (printable-repr (car pair))))))

;;; ---------- Main entry point ----------

(defun analyze-chars (corpus-path)
  (let* ((text (load-corpus corpus-path))
         (freq (char-frequencies text)))
    (format t "Corpus: ~A~%" corpus-path)
    (print-summary text freq)
    (values)))

(analyze-chars "corpus/poeall.txt")
