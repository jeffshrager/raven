(:MODEL
 (:CORPUS-PATH "corpus/poeall.txt" :D-MODEL 64 :N-HEADS 4 :N-LAYERS 3 :CONTEXT
  64 :LR 0.001 :STEPS (:SCAN 1000 2000 3000 4000))
 :TESTS :ALL :DESCRIPTION
 "Same architecture again, no new step counts -- every value should already be :completed, so this should train nothing at all and just re-score the existing checkpoints.")
