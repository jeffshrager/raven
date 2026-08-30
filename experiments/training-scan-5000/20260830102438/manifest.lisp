(:MODEL
 (:CORPUS-PATH "corpus/poeall.txt" :D-MODEL 64 :N-HEADS 4 :N-LAYERS 3 :CONTEXT
  64 :LR 0.001 :STEPS (:SCAN 1000 2000 3000 4000 5000))
 :TESTS :ALL :DESCRIPTION
 "Same architecture as training-scan, extended to 5000 steps -- 1000/2000/3000/4000 should be reused from the registry (no retraining), only 5000 should train fresh.")
