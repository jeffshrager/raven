;;; (load (compile-file "scratch/token-analysis.lisp"))
;;;
;;; Word-level token analysis for a text corpus.
;;; Reports vocabulary size, frequency distribution, coverage curves,
;;; and hapax count. Runs both case-sensitive and case-folded.

;;; ---------- File loading ----------

(defun load-corpus (path)
  "Read entire file at PATH into a single string."
  (with-open-file (stream path :direction :input)
    (let ((contents (make-string (file-length stream))))
      (read-sequence contents stream)
      contents)))

;;; ---------- Tokenization ----------
;;; Strategy: split on whitespace, but separate punctuation from words.
;;; So "Nevermore!" becomes ("Nevermore" "!").
;;; Punctuation chars each become their own token.

(defparameter *punctuation-chars*
  '(#\. #\, #\; #\: #\! #\? #\" #\' #\( #\) #\[ #\] #\- #\— #\newline))

(defun punctuation-char-p (c)
  (member c *punctuation-chars* :test #'char=))

(defun whitespace-char-p (c)
  (or (char= c #\Space) (char= c #\Tab) (char= c #\Return)))

(defun tokenize-words (text)
  "Split TEXT into word and punctuation tokens.
   Whitespace is discarded; newlines become their own tokens
   (they carry structural meaning in a poem)."
  (let ((tokens '())
        (current (make-string-output-stream)))
    (labels ((flush ()
               (let ((s (get-output-stream-string current)))
                 (when (> (length s) 0)
                   (push s tokens))
                 (setf current (make-string-output-stream)))))
      (loop for c across text do
        (cond
          ((whitespace-char-p c)
           (flush))
          ((punctuation-char-p c)
           (flush)
           (push (string c) tokens))
          (t
           (write-char c current))))
      (flush))
    (nreverse tokens)))

;;; ---------- Frequency counting ----------

(defun count-tokens (tokens)
  "Return an alist of (token . count), sorted by count descending."
  (let ((table (make-hash-table :test #'equal)))
    (dolist (tok tokens)
      (incf (gethash tok table 0)))
    (let ((pairs '()))
      (maphash (lambda (k v) (push (cons k v) pairs)) table)
      (sort pairs #'> :key #'cdr))))

;;; ---------- Coverage analysis ----------

(defun coverage-curve (freq-alist thresholds)
  "For each threshold in THRESHOLDS (fractions 0..1), return an alist of
   (threshold . number-of-top-tokens-needed-to-cover-that-fraction)."
  (let* ((total (reduce #'+ freq-alist :key #'cdr))
         (results '()))
    (dolist (thresh thresholds)
      (let ((target (* thresh total))
            (running 0)
            (count 0))
        (loop for (tok . n) in freq-alist
              do (incf running n)
                 (incf count)
              until (>= running target))
        (push (cons thresh count) results)))
    (nreverse results)))

(defun hapax-count (freq-alist)
  "Number of tokens appearing exactly once."
  (count 1 freq-alist :key #'cdr))

;;; ---------- Reporting ----------

(defun print-summary (label freq-alist total-tokens)
  (let ((unique (length freq-alist))
        (hapaxes (hapax-count freq-alist))
        (coverage (coverage-curve freq-alist '(0.5 0.8 0.9 0.95 0.99))))
    (format t "~%=== ~A ===~%" label)
    (format t "Total tokens:  ~D~%" total-tokens)
    (format t "Unique tokens: ~D~%" unique)
    (format t "Hapaxes (freq=1): ~D (~,1F% of vocab)~%"
            hapaxes (* 100.0 (/ hapaxes unique)))
    (format t "Compression:   ~,2F tokens per unique~%"
            (/ total-tokens unique))
    (format t "Coverage curve:~%")
    (dolist (pair coverage)
      (format t "  ~,0F%% covered by top ~D tokens (~,1F%% of vocab)~%"
              (* 100 (car pair))
              (cdr pair)
              (* 100.0 (/ (cdr pair) unique))))
    (format t "Top 20 tokens:~%")
    (loop for (tok . n) in freq-alist
          for i from 1 to 20
          do (format t "  ~2D. ~15S ~5D~%" i tok n))))

(defun write-full-frequencies (path freq-alist)
  "Write the entire frequency table to PATH, one line per token."
  (with-open-file (out path :direction :output
                            :if-exists :supersede
                            :if-does-not-exist :create)
    (dolist (pair freq-alist)
      (format out "~6D  ~S~%" (cdr pair) (car pair)))))

;;; ---------- Main entry point ----------

(defun analyze-corpus (corpus-path &key (log-dir "results/logs/"))
  (let* ((text (load-corpus corpus-path))
         (tokens-cs (tokenize-words text))
         (tokens-cf (mapcar #'string-downcase tokens-cs))
         (freq-cs (count-tokens tokens-cs))
         (freq-cf (count-tokens tokens-cf)))
    (format t "Corpus: ~A~%" corpus-path)
    (format t "Raw length: ~D characters~%" (length text))
    (print-summary "CASE-SENSITIVE" freq-cs (length tokens-cs))
    (print-summary "CASE-FOLDED"    freq-cf (length tokens-cf))
    (write-full-frequencies
      (concatenate 'string log-dir "freq-case-sensitive.txt") freq-cs)
    (write-full-frequencies
      (concatenate 'string log-dir "freq-case-folded.txt") freq-cf)
    (format t "~%Full frequency tables written to ~A~%" log-dir)
    (values freq-cs freq-cf)))

#(analyze-corpus "corpus/raven.txt")
(analyze-corpus "corpus/poeall.txt")
