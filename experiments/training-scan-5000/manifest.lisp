(:model (:corpus-path "corpus/poeall.txt"
         :d-model  64
         :n-heads  4
         :n-layers 3
         :context  64
         :lr       0.001
         :steps    (:scan 1000 2000 3000 4000 5000))
 :tests :all
 :description "Same architecture as training-scan, extended to 5000 steps -- 1000/2000/3000/4000 should be reused from the registry (no retraining), only 5000 should train fresh.")
