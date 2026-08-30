# A Hacker's Guide to Raven-LLM

*For the curious visitor who wants to see a transformer language model with nothing hidden underneath it*

*Written in the style of [Hacker's Guides](https://github.com/jeffshrager/HackersGuides) — close readings of landmark source code, for the curious visitor rather than the specialist. This guide isn't part of that collection (Raven is a live, ongoing project, not a historic artifact), but borrows its format: a map, a tour through the districts, and the details worth a special trip.*

---

## Before You Arrive: What Is Raven?

Raven is a char-level GPT-style transformer, written from scratch in Common Lisp (SBCL), begun and actively developed in 2026. It is not a historic artifact — the guide you're reading may be out of date by the time you read it, and that's fine; treat this as a map of a real, moving place rather than a monument.

What makes Raven worth touring isn't novelty of result — it produces the same kind of output any small char-level GPT produces — but transparency of mechanism. There is no PyTorch, no autograd, no BLAS. Every matrix multiply, every softmax, every gradient is a hand-written loop over `(simple-array single-float (...))` tensors, with the backward pass for each operation written out by hand next to its forward pass. If you have ever wondered what "backpropagation" actually computes, line by line, this is a codebase small enough (about 4,200 lines across eleven files) to read start to finish in an afternoon and come away able to derive the rest yourself.

It runs as an ASDF system under SBCL. There is no GPU, no batching — one sequence at a time, trained on plain CPU loops. This is a deliberate trade: Raven is legible in exchange for being slow.

---

## The Map: Two Halves

Raven splits cleanly into two concerns that don't know about each other until the very top: **the model** (how a transformer computes and learns) and **the lab** (how you run it as an actual experimental practice — training, checkpointing, comparing runs). A training step flows through the first; a research session flows through the second.

```
                     THE MODEL                              THE LAB
tensors  →  ops  →  modules  →  composition  →  training  →  experiment
[utilities] [tensor-ops] [modules] [model.lisp] [training.lisp] [experiment.lisp]
                                       ↑                              ↑
                                  [gradcheck]                  [data, checkpoint, inference]
```

Everything in "the model" column is loaded by `raven.asd` in exactly that dependency order — each file only knows about the ones before it. `experiment.lisp` sits at the top of the stack and is the only file that talks to the filesystem in any structured way (models, manifests, logs). There's no separate driver script; `raven.asd`'s load order *is* the architecture diagram.

---

## District 1: The Ground Floor — `utilities.lisp`, `tensor-ops.lisp`

Before anything resembling a neural network appears, two files establish the physics everything else stands on.

`utilities.lisp` is sixteen lines of load-corpus gotcha (a UTF-8 file's byte count and character count differ, so a naive `make-string (file-length ...)` allocates the wrong size and leaves trailing NULs — the fix is to trust `read-sequence`'s return value, not the file length) plus the tensor constructors and Xavier initialization every weight matrix is built from.

> **★ DON'T MISS**
>
> `xavier-init`'s parameters are named `d-in`/`d-out`, not the textbook `fan-out`/`fan-in`. This isn't a stylistic quirk — it's a project-wide convention stated in `modules.lisp`'s header and load-bearing everywhere: Raven stores every weight matrix as `(d_in, d_out)` so `matmul-forward(input, W)` works with no transpose, the opposite of the classical linear-algebra `y = Wx` column-vector convention. If you've internalized Xavier init from a textbook or another codebase, the axis order here will look backwards until you notice this. (There actually *was* a stale docstring describing the wrong convention here for a while — a small proof that even a from-scratch, hand-audited codebase accumulates the ordinary kind of drift.)

`tensor-ops.lisp` is where the actual math lives: `matmul`, `softmax`, `gelu`, `add-bias`, `transpose`, `embedding-lookup` — nine primitive operations, each one a matched *forward* and *backward* function pair. The convention, stated once in the file header and then followed without exception: forward returns `(values output saved)`, backward takes `(grad-out saved)` and returns gradients for every input, and **every gradient tensor has the same shape as the value it's a gradient for.** Once you know that invariant, you can read any `*-backward` function's `declare` forms and know exactly what it should produce before you've read the body.

> **★ DON'T MISS**
>
> `softmax-forward`'s row-max subtraction. The comment explains *why* it's safe (softmax is invariant under adding a constant to every input, so subtracting the row's own max just prevents `exp()` overflow) rather than just asserting that it's done — the kind of thing that's easy to copy as boilerplate and much more useful once you've seen the one-line proof that it doesn't change the answer.

---

## District 2: The Assembly Floor — `modules.lisp`

This is the largest file and the one most worth lingering in. It defines a tiny plugin registry — every module type (`:embedding`, `:rmsnorm`, `:attention-block`, `:ffn`, `:transformer-block`, `:unembedding`) registers an allocator, a forward, and a backward function under a keyword, and everything downstream dispatches through that registry rather than calling any module directly by name.

```lisp
(defun module-forward (type params input context)
  (funcall (getf (get-module type) :forward) params input context))
```

That's the entire dispatch mechanism. Adding a new module type — a different normalization, a different position encoding, RoPE instead of sinusoidal — is one new file and one `register-module` call; nothing else in the codebase needs to know it exists.

> **★ DON'T MISS**
>
> The comment in `attention-fwd` walking through a concrete T=4 causal mask by hand:
> ```
> row i=0: [ S00  -inf -inf -inf ]   attends to {0}
> row i=1: [ S10   S11 -inf -inf ]   attends to {0,1}
> row i=2: [ S20   S21  S22 -inf ]   attends to {0,1,2}
> row i=3: [ S30   S31  S32  S33 ]   attends to {0,1,2,3}
> ```
> This is the entire mechanism by which a GPT is *causal* — a next-token predictor that can't see the future — reduced to one condition, `j > i`, on one loop. No separate "masking layer," no special data structure: just don't let a query attend to a key that comes later in the sequence.

> **★ DON'T MISS**
>
> `slice-head`/`scatter-head!`, the pair of functions that split a `(T, D)` tensor into `H` independent `(T, dk)` head-subspaces and reassemble them afterward. Multi-head attention is often explained as though the heads are somehow separate all along; here you can see plainly that they're just contiguous column-blocks of one wider matrix, sliced out before each head does its own attention and scattered back before the output projection. The worked D=8/H=4 example in the comment makes the column arithmetic (`start = head-idx * dk`) concrete rather than asserted.

> **★ DON'T MISS**
>
> The transformer block is explicitly **pre-norm** (RMSNorm applied to a sub-layer's *input*, with the residual add carrying the *un-normalized* stream forward), and the backward-pass comment explains exactly why that matters for training depth: the residual connections give the gradient a path that never passes through attention or the FFN at all. `grad-input`'s contribution from the residual add is literally never touched by `module-backward :attention-block` — you can watch that non-dependency in the code, not just take it on faith from a paper.

---

## District 3: Learning — `training.lisp`, `gradcheck.lisp`

`training.lisp` fuses softmax and cross-entropy into one numerically stable operation (`softmax-ce-fwd`/`-bwd`) and implements Adam from the 2014 paper directly — no framework, no `torch.optim`, just the update rule:

```lisp
(decf (row-major-aref p i)
      (/ (* lr mhat) (+ (sqrt vhat) eps)))
```

> **★ DON'T MISS**
>
> `softmax-ce-bwd`'s comment gives an actual worked example with numbers — probs `[0.2, 0.5, 0.3]`, target index 1, gradient `[0.2, -0.5, 0.3]` — and reads off the direction of each sign: negative at the target (push that logit *up*), positive everywhere else (push those logits *down*). This is the entire "reward the right answer, penalize the rest" intuition behind cross-entropy training, visible as three numbers rather than an abstract formula.

`collect-tensors` is the unglamorous function that makes the whole optimizer generic: it walks a nested params plist (transformer blocks nest inside the top-level model, RMSNorm and attention nest inside each block) and flattens every trainable tensor into one deterministic list, skipping the fixed sinusoidal position table by name (`:pe`) so it never receives or needs a gradient. Adam, checkpoint save/load, and gradient checking all call this same walker — they never know or care about the model's actual nesting depth.

`gradcheck.lisp` is the project's honesty mechanism: every module's hand-derived backward pass is checked against a numerical (central-difference) approximation of the same gradient, with combined absolute+relative tolerances borrowed from `torch.autograd.gradcheck`'s convention.

> **★ DON'T MISS**
>
> The file header's derivation of *why* the check is valid at all: pick a random upstream gradient `U`, define a scalar loss `L = sum(output * U)`, and by construction `dL/dparam` equals exactly what `backward(U)` computes. This is the standard trick for gradient-checking any differentiable function, stated here in five lines rather than assumed. If you've never seen why gradcheck works, not just that it's a thing people do, it's here.

---

## District 4: The Lab — `experiment.lisp`

Everything above this file is a self-contained numerical library — it would work exactly the same way as a REPL toy. `experiment.lisp` is where Raven becomes a *practice*: a way of running many training configurations, keeping honest records of what happened, and never doing the same expensive work twice by accident.

The central idea is that **a model is identified by what it is, not by a name you picked for it.** There's no `train("my-model-v3")`. Instead:

```lisp
(ensure-model '(:corpus-path "corpus/poeall.txt" :d-model 64 :n-layers 3 :steps 5000))
```

`ensure-model` canonicalizes that spec (fills in every default, sorts the keys) and checks a registry file, `models/directory.lisp`, for an existing model whose canonical spec is `equal`. If one exists — and only if it reached `:completed` status — it's reused instantly, no training. If not, it trains one, and only then registers it.

> **★ DON'T MISS**
>
> The three-state lifecycle every model passes through: `:incomplete` (written to the registry *before* training starts) → `:completed` or `:failed`. This exists because of a real failure mode: SHRDLU-era or not, any long-running process can die mid-run, and a naive "save the checkpoint when done" design leaves an orphaned file with zero trace of what happened. Here, a crash after a mid-run auto-checkpoint has already been written still leaves an honest `:incomplete` record — and `find-model-by-spec` refuses to ever reuse anything that isn't `:completed`, so a broken attempt can never masquerade as a finished model on the next run.

> **★ DON'T MISS**
>
> The `(:scan v1 v2 ...)` sweep syntax in an experiment manifest, and its one hand-optimized special case: scanning `:steps` doesn't train each value independently. Since `(:steps (:scan 1000 2000 3000 4000))` describes models that are literal prefixes of one another under identical Adam updates, `train-with-milestones` runs training *once*, to the largest value, and registers a fully independent `:completed` model at every requested step count along the way via a generic `on-milestone` callback threaded through the step loop. A four-value scan costs 4,000 training steps, not 10,000. This was suggested mid-session by watching the pattern in a running experiment, not planned in advance — the kind of optimization that's obvious once you're staring at the actual cost, and easy to miss from a spec.

---

## A Day in the Life: One Sentence, End to End

Trace what happens when `train-step!` is called once, with the token sequence for `"the raven"` as input:

**Step 1 — `model-fwd`** threads the token ids through the spec in order: `:embedding` looks up each id's row in the embedding table `(T, D)`; `:positional-sinusoidal` adds a fixed sin/cos position code; each `:transformer-block` runs pre-norm attention-then-residual, pre-norm FFN-then-residual; `:final-norm` normalizes once more; `:unembedding` projects to `(T, V)` raw logits — one score per position per possible next character.

**Step 2 — `softmax-ce-fwd`** computes the mean per-token negative log-likelihood of the *actual* next characters (the same sequence shifted by one) against those logits, using the numerically stable logsumexp trick, and saves the softmax probabilities.

**Step 3 — `softmax-ce-bwd`** turns that scalar loss back into a `(T, V)` gradient — for the correct next-character at each position, a negative push; for every other character, a positive push (see the worked example in District 3).

**Step 4 — `model-bwd`** walks the same spec in reverse, calling each module's own hand-derived backward function, accumulating gradients wherever a value was used more than once (the residual streams, the embedding table when a character repeats, RMSNorm's shared gamma).

**Step 5 — `adam-step!`** applies every accumulated gradient to its parameter, in place, using bias-corrected first and second moment estimates.

Five steps, no framework, and every one of them is under a hundred lines you can actually read.

---

## Practical Notes for the Independent Traveler

**On Common Lisp, if you're coming from elsewhere.** `defstruct` is a lightweight record type. `getf`/`(getf plist :key default)` reads a plist (a flat list alternating keys and values — Raven's universal container for both hyperparameter configs and model parameters). `multiple-value-bind` destructures a function's extra return values (Lisp functions can return more than one value without boxing them in a tuple). `declare (type ...)` is a compiler hint, not a runtime check — SBCL uses these aggressively to generate tight numeric code, which is most of why plain nested loops are fast enough here without SIMD or BLAS.

**On the two "spec" words.** Two different things in this codebase are called "spec," and the header comments say so explicitly precisely because it's an easy mix-up: a **training-spec** is the user-facing plist (`:corpus-path`, `:d-model`, `:steps`, ...) that identifies a model in the registry; a **module-spec** is the lower-level list of `(name type config)` triples that `build-gpt-lm-spec` produces and `model-fwd`/`model-bwd` actually execute. If a function's argument is named `spec`, check which file you're in.

**On running it yourself.** `README.md` has the full walkthrough. The short version: `sbcl --load raven.asd --eval '(asdf:load-system :raven)' --eval '(test-jig)'` runs an end-to-end synthetic check (no corpus required) that exercises training, the model registry, sweeps, and reuse all in one call, and reports `PASS`/`FAIL`.

**On what's genuinely unfinished.** There's no batching (one sequence per step), no KV-cache for generation (`generate` reprocesses the whole sliding window every token), and no resume-from-checkpoint (loading an existing model's weights and continuing training from them) — the last was considered and deliberately shelved, because Adam's optimizer state isn't persisted in the checkpoint format, so a "resumed" run's momentum would restart from zero partway through: a genuinely different optimization trajectory from a continuous run of the same nominal step count, not just a shortcut to the same place.

---

## What Makes Raven Worth Visiting

Raven's interest isn't in doing anything a bigger, faster, framework-backed model doesn't already do better. It's that nothing here is a black box you're asked to trust. The backward pass for causal attention is forty lines you can step through by hand. The Adam update is the paper's four-line formula and nothing else. The file that decides whether to retrain a model or reuse one is a single `equal` comparison on a canonicalized plist.

That legibility comes at a real cost — this is slow, unbatched, single-sequence CPU code, and it will never train anything you'd call large. But the trade is the point: every mechanism a modern LLM training stack hides behind an optimized kernel or a framework abstraction is, here, a loop you can read, a shape you can trace, and — via `gradcheck.lisp` — a claim you can verify rather than take on faith.

---

*This guide describes Raven-LLM as of the commits discussed in its `seshsums/` session logs. Being a live project, both the code and this guide will drift — check `git log` and the `seshsums/` directory for what's changed since. For more guides in this style, covering programs that made history rather than ones still making it, see [Hacker's Guides](https://github.com/jeffshrager/HackersGuides).*
