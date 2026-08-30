(:model (:corpus-path "corpus/poeall.txt"
         :d-model  64
         :n-heads  4
         :n-layers 3
         :context  64
         :lr       0.001
         :steps    (:scan 1000 2000 3000 4000))
 :tests :all
 :description "Same architecture again, no new step counts -- every value should already be :completed, so this should train nothing at all and just re-score the existing checkpoints.")
