;;;; raven.asd
;;;;
;;;; ASDF system definition for Raven-LLM.
;;;;
;;;; Usage (from project root):
;;;;   sbcl --load raven.asd \
;;;;        --eval '(asdf:load-system :raven)' \
;;;;        --eval '(train-and-register "poe-64d-3l")'
;;;;
;;;; Or interactively:
;;;;   (load "raven.asd")
;;;;   (asdf:load-system :raven)
;;;;
;;;; ASDF handles dependency tracking and only rebuilds files whose
;;;; source is newer than their fasl. Everything is loaded into the
;;;; :cl-user package (this project doesn't use its own package).

;; SBCL ships with ASDF but doesn't preload it for plain --load of a
;; .asd file. Force it in so DEFSYSTEM is available.
(require "asdf")

(asdf:defsystem :raven
  :description "Raven-LLM: a minimal char-level LLM from scratch in Common Lisp."
  :version     "0.1"
  :serial      t
  :pathname    "src"
  :components  ((:file "utilities")
                (:file "tensor-ops")
                (:file "modules")
                (:file "gradcheck")
                (:file "training")
                (:file "model")
                (:file "data")
                (:file "inference")
                (:file "checkpoint")
                (:file "experiment")))
