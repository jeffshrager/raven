(defun load-corpus (path)
  "Read entire file at PATH into a single string. Uses read-sequence's
   return value to handle the case where char count < byte count
   (multi-byte UTF-8 makes file-length an overestimate of char count)."
  (with-open-file (stream path :direction :input)
    (let* ((buffer (make-string (file-length stream)))
           (actual (read-sequence buffer stream)))
      (subseq buffer 0 actual))))
