(:MODEL
 (:CORPUS-PATH "corpus/poeall.txt" :D-MODEL 64 :N-HEADS 4 :N-LAYERS 3 :CONTEXT
  64 :LR 0.001 :STEPS (:SCAN 1000 2000 3000 4000))
 :TESTS :ALL :DESCRIPTION
 "Training-length scan: how much does final loss / test performance improve going from 1000 to 4000 steps, at a fixed architecture (d=64, H=4, L=3, T=64) on Poe's complete works?")
