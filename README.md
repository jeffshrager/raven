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
│   └── experiment.lisp    #   model registry (ensure-model), run-experiment, test-jig
├── models/
│   ├── directory.lisp     #   registry: one entry per model attempt (spec, id, checkpoint, status)
│   ├── build.log          #   append-only audit trail (:trained / :evaluated events)
│   └── <id>.ravn          #   checkpoint for model <id> (timestamped, not user-named)
├── tests/                 # challenge files: NAME.txt, each line "PROMPT|EXPECTED"
├── experiments/
│   └── <name>/
│       ├── manifest.lisp  #   the experiment recipe (plist, user-written)
│       ├── index.log      #   append-only, one line per run dir ever created here
│       └── <timestamp>/   #   one per concrete model tested; NEVER named from a parameter value
├── corpus/                # training text
└── seshsums/              # human-readable dev session summaries, one per session
```

## Quick verification (no corpus needed)

```bash
sbcl --load raven.asd \
     --eval '(asdf:load-system :raven)' \
     --eval '(test-jig)'
```

This runs an experiment that scans a training-spec parameter across two values on a synthetic repeating string — training two tiny models, running both through a mini test suite, then re-running the same experiment to prove the models get reused rather than retrained — and reports `PASS` if every expected file and registry entry is in the right state.

## Complete example

### 1. Put a corpus in place

```bash
mkdir -p corpus
# any UTF-8 text file will do
curl -o corpus/poeall.txt https://www.gutenberg.org/cache/epub/2148/pg2148.txt
```

### 2. Models are identified by their spec, not a name

There's no `.model` file to write by hand and no name to pick. A model is a **training-spec** plist — the same shape a `.model` file used to hold:

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

Every key except `:corpus-path` is optional (defaults in the docstring of `canonicalize-training-spec` in `src/experiment.lisp`). You normally never call the registry directly — it's driven by an experiment manifest (step 3) — but you can build one ad hoc:

```lisp
(ensure-model '(:corpus-path "corpus/poeall.txt" :d-model 64 :n-layers 3 :steps 5000))
```

`ensure-model` canonicalizes the spec (fills in defaults, sorts keys) and checks `models/directory.lisp` for a **:completed** model with an exactly matching spec. If one exists, it's reused instantly — no retraining. Otherwise a fresh model is trained and registered under a new timestamped id (`models/<id>.ravn`).

Before training starts, an entry is written with `:status :incomplete` — so if the process dies mid-run (crash, kill, power loss), even after a mid-run auto-checkpoint has been saved, the registry shows an honest `:incomplete` record instead of an orphaned, untracked `.ravn` file. That same entry is updated in place once training ends: to `:completed` (with the final loss) on success, or to `:failed` (with the error text) if training raised an error — the error still propagates to the caller either way. A spec whose only prior attempt is `:incomplete` or `:failed` is never reused; asking for it again always starts a fresh attempt under a new id. Every state transition is also appended to `models/build.log` (`:training-started` / `:trained` / `:training-failed` / `:evaluated`) as a flat, human-readable audit trail.

Loss is printed periodically during training. By default, the checkpoint is also saved mid-run every 1000 steps (overwriting each time), so a long or interrupted run still leaves a usable checkpoint behind — set `:checkpoint-every` in the spec to change the interval, or `nil` to disable it. This doesn't affect the registry entry, only how the run behaves — training-spec keys that don't define the model itself (`:checkpoint-every`, `:log-every`, `:eval-samples`, `:description`) are ignored when matching against existing models.

### 3. Use the included test file

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

### 4. Write an experiment manifest

Create `experiments/eval-1/manifest.lisp`. `:model` is a training-spec (see step 2) — train/reuse one model and test it:

```lisp
(:model (:corpus-path "corpus/poeall.txt" :d-model 64 :n-layers 3 :steps 5000)
 :tests ("test1.txt")
 :description "first evaluation on The Raven excerpts")
```

Or scan any number of spec keys with `(:scan v1 v2 ...)` — every combination is trained-or-reused and tested (a cartesian product, so scanning two keys with three values each tests nine models):

```lisp
(:model (:corpus-path "corpus/poeall.txt" :d-model 64 :n-layers (:scan 2 3 4) :steps 5000)
 :tests :all
 :description "layer-depth sweep")
```

Scanning `:steps` is special-cased for efficiency: since every value shares the same architecture/corpus/lr and differs only in how long training ran, a single continuous run for the *largest* value passes through exactly the state a standalone run for each smaller value would have stopped at. So `(:steps (:scan 1000 2000 3000 4000))` trains once for 4000 steps and registers a real, independently `:completed` model at each of the four step counts along the way — 4000 total training steps, not 1000+2000+3000+4000=10000. (If some of those step counts are already `:completed` from an earlier run, only the still-missing ones get trained, in one shot up to the largest missing value.) This kicks in automatically whenever `:steps` is scanned, alone or alongside other scanned keys.

Or name one exact, already-registered model by its id to re-run tests against it without ever retraining:

```lisp
(:model "20260830101512" :tests ("test1.txt"))
```

`:tests` can also be `:all` to run every `tests/*.txt` file.

### 5. Run the experiment

```bash
sbcl --load raven.asd \
     --eval '(asdf:load-system :raven)' \
     --eval '(run-experiment "eval-1")'
```

For each concrete model resolved from `:model` (one, unless scanning), this creates a freshly timestamped `experiments/eval-1/<timestamp>/` — subdirectory names are always timestamps, never derived from a scanned value — containing:

| File | Contents |
|------|----------|
| `manifest.lisp`    | copy of the manifest for provenance |
| `model-spec.lisp`  | which model this directory's results are for: id, checkpoint path, resolved spec |
| `results.log`      | one alist per challenge — prompt, expected, greedy generation, NLL, prefix-match, per-char accuracy, tagged `:model-id` |
| `summary.log`      | aggregate stats + per-test breakdown for that model |

`experiments/eval-1/index.log` also gets one line per run directory ever created there (across every invocation), so you can see a sweep's headline numbers without opening each subdirectory. `models/build.log` picks up an `:EVALUATED` entry per model tested, pointing back at the experiment name and run timestamp.

## Scoring

For each challenge the jig computes:

- **NLL** — mean per-token negative log-likelihood of the expected continuation given the prompt. Deterministic, doesn't sample. This is the primary quality metric — it's the standard LM eval and matches the training objective.
- **Greedy generation** — what the model actually produces (argmax at each step) for a qualitative look.
- **Prefix-match length** — how many leading characters of the greedy output agree with expected.
- **Per-char accuracy** — fraction of matching characters across the whole span.

## File formats

All persisted metadata is S-expressions — no JSON anywhere.

- User-written config (a training-spec, an experiment manifest): **plist** — `(:key value :key value ...)`
- Everything else (auto-generated metadata: `directory.lisp`, `build.log`, `model-spec.lisp`, `results.log`, `summary.log`, `index.log`): **alist** — `((:key . value) ...)`

Both round-trip through `read`.

## Interactive use

Everything works interactively too:

```lisp
(load "raven.asd")
(asdf:load-system :raven)
(ensure-model '(:corpus-path "corpus/poeall.txt" :d-model 64 :n-layers 3 :steps 5000))
(run-experiment "eval-1")

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
