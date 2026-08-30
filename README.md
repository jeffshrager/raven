# Raven-LLM

A minimal char-level transformer language model, from-scratch in Common Lisp (SBCL). Hand-written backprop, no external ML dependencies.

## Requirements

- SBCL (tested with 2.2.9)
- A UTF-8 text corpus to train on

## Project layout

```
raven/
├── raven.asd              # ASDF system definition
├── src/                   # source files (loaded in this order)
│   ├── utilities.lisp     #   tensor primitives + corpus loader
│   ├── tensor-ops.lisp    #   matmul, softmax, gelu (+ backwards)
│   ├── modules.lisp       #   embedding, RMSNorm, attention, FFN, block, unembedding
│   ├── gradcheck.lisp     #   numerical gradient checking
│   ├── training.lisp      #   fused softmax-CE loss + Adam
│   ├── model.lisp         #   spec-based composition (build-gpt-lm-spec, model-fwd, model-bwd)
│   ├── data.lisp          #   vocab, encode/decode, window sampler
│   ├── inference.lisp     #   sampling (greedy/temperature/top-k/top-p) + generate
│   ├── checkpoint.lisp    #   binary save/load
│   └── experiment.lisp    #   train-and-register, run-experiment, test-jig
├── models/                # trained models: NAME.model / .ravn / .manifest / .log
├── tests/                 # challenge files: NAME.txt, each line "PROMPT|EXPECTED"
├── experiments/           # timestamped run outputs
└── corpus/                # training text
```

## Quick verification (no corpus needed)

```bash
sbcl --load raven.asd \
     --eval '(asdf:load-system :raven)' \
     --eval '(test-jig)'
```

This trains a tiny model on a synthetic repeating string, saves a checkpoint, runs a mini experiment against it, and reports `PASS` if all expected output files were produced.

## Complete example

### 1. Put a corpus in place

```bash
mkdir -p corpus
# any UTF-8 text file will do
curl -o corpus/poeall.txt https://www.gutenberg.org/cache/epub/2148/pg2148.txt
```

### 2. Write a model spec

Create `models/poe-64d-3l.model` (any name; the filename before `.model` becomes the model's identifier):

```lisp
(:corpus-path "corpus/poeall.txt"
 :d-model     64
 :n-heads     4
 :n-layers    3
 :context     64
 :steps       5000
 :lr          0.001
 :description "3-layer 64-dim GPT trained on Poe complete works")
```

Every key except `:corpus-path` is optional. Full list of supported keys and defaults is in the docstring of `train-and-register` in `src/experiment.lisp`.

### 3. Train

```bash
sbcl --load raven.asd \
     --eval '(asdf:load-system :raven)' \
     --eval '(train-and-register "poe-64d-3l")'
```

Loss is printed periodically. When training finishes, four files exist in `models/`:

| File | Purpose |
|------|---------|
| `poe-64d-3l.model`    | your spec (unchanged) |
| `poe-64d-3l.ravn`     | binary checkpoint — params + spec + vocab |
| `poe-64d-3l.manifest` | metadata alist: config, final loss, timestamp |
| `poe-64d-3l.log`      | append-only event log (one alist per line) |

Re-running `train-and-register` overwrites the `.ravn` and `.manifest` but appends to `.log`, so retraining history is preserved.

### 4. Use the included test file

The repo ships with `tests/test1.txt` — nine challenges drawn from *The Raven*. Excerpt:

```
; ---- short word completions ----
Once upon a midnight dr|eary
Deep into that darkness peer|ing

; ---- longer contextual completions ----
Perched upon a bust of Pallas |just above my chamber door
```

Format is one challenge per line, `|` splitting prompt from expected. Lines starting with `;` are comments; blank lines are ignored.

Constraint: prompt + expected must fit in the model's `:context`. Longer lines are recorded as skipped in `results.log` (the last challenge in `test1.txt` deliberately overflows the default 64-char context so you can see how the skip appears).

To add your own tests, just drop another `tests/*.txt` file alongside it.

### 5. Write an experiment manifest

Create `experiments/eval-1.manifest`:

```lisp
((:model . "poe-64d-3l")
 (:tests . ("test1.txt"))
 (:description . "first evaluation of poe-64d-3l on The Raven excerpts"))
```

The `:tests` value can also be `:all` to run every `tests/*.txt` file.

### 6. Run the experiment

```bash
sbcl --load raven.asd \
     --eval '(asdf:load-system :raven)' \
     --eval '(run-experiment "experiments/eval-1.manifest")'
```

This creates `experiments/<timestamp>/` containing:

| File | Contents |
|------|----------|
| `manifest.lisp` | copy of the manifest for provenance |
| `results.log`   | one alist per challenge — prompt, expected, greedy generation, NLL, prefix-match, per-char accuracy |
| `summary.log`   | aggregate stats + per-test breakdown |

The model's log picks up an `:EVALUATED` entry pointing back to the experiment timestamp.

## Scoring

For each challenge the jig computes:

- **NLL** — mean per-token negative log-likelihood of the expected continuation given the prompt. Deterministic, doesn't sample. This is the primary quality metric — it's the standard LM eval and matches the training objective.
- **Greedy generation** — what the model actually produces (argmax at each step) for a qualitative look.
- **Prefix-match length** — how many leading characters of the greedy output agree with expected.
- **Per-char accuracy** — fraction of matching characters across the whole span.

## File formats

All persisted metadata is S-expressions — no JSON anywhere.

- `.model` files (user-written config): **plist** — `(:key value :key value ...)`
- Everything else (auto-generated metadata: `.manifest`, `.log`, `results.log`, `summary.log`): **alist** — `((:key . value) ...)`

Both round-trip through `read`.

## Interactive use

Everything works interactively too:

```lisp
(load "raven.asd")
(asdf:load-system :raven)
(train-and-register "poe-64d-3l")
(run-experiment "experiments/eval-1.manifest")

;; Or train ad-hoc without registering:
(multiple-value-bind (params spec vocab)
    (train :corpus-path "corpus/poeall.txt" :steps 1000)
  (format t "~A~%" (generate params spec vocab "The raven "
                             :n-tokens 200 :top-k 40 :temperature 0.8)))
```

## Individual module tests

Each source file has its own sanity check:

```lisp
(run-all-gradchecks)           ; src/gradcheck.lisp
(run-training-tests)           ; src/training.lisp
(run-data-tests)               ; src/data.lisp
(test-overfit-single-sequence) ; src/model.lisp
(test-generate)                ; src/inference.lisp
(test-checkpoint)              ; src/checkpoint.lisp
(test-jig)                     ; src/experiment.lisp
```
