;;;; main.lisp — thin compatibility shim.
;;;;
;;;; The project now uses ASDF. Preferred incantation (from repo root):
;;;;   sbcl --load raven.asd \
;;;;        --eval '(asdf:load-system :raven)' \
;;;;        --eval '(train-and-register "poe-64d-3l")'
;;;;
;;;; This file exists only so `sbcl --load src/main.lisp` still works —
;;;; it loads the ASDF system and runs the DEMO. Everything that used
;;;; to be defined here (TRAIN, DEMO, TRAIN-LOOP, LOSS-LOG) now lives
;;;; in src/experiment.lisp, at the top of the ASDF chain.

(load "raven.asd")
(asdf:load-system :raven)
(demo)
