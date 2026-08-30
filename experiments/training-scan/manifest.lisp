(:model (:corpus-path "corpus/poeall.txt"
         :d-model  64
         :n-heads  4
         :n-layers 3
         :context  64
         :lr       0.001
         :steps    (:scan 1000 2000 3000 4000))
 :tests :all
 :description "Training-length scan: how much does final loss / test performance improve going from 1000 to 4000 steps, at a fixed architecture (d=64, H=4, L=3, T=64) on Poe's complete works?")
