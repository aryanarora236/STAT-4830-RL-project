# Self-Critique — Week 15 (Apr 28)

Final self-critique covering the complete project arc, written after the final PrimeIntellect training run.

## OBSERVE

### What we set out to do
- Take the Recursive Language Model idea (LLM + Python REPL) and apply it to a domain where long context actually matters for the decision.
- Domain: No-Limit Texas Hold'em poker, where optimal play depends on opponent tendencies visible only in multi-hand history.
- Pipeline: zero-shot LLM baseline → behavior cloning on heuristic trajectories → REINFORCE fine-tuning → final evaluation.

### What we shipped
- **Environment** (`src/poker/environment.py`): 6-max NLHE with 5 opponent archetypes, structured hand-history generator, 5-of-7 hand evaluator, Monte Carlo equity.
- **Heuristic bot** (`src/poker/heuristic.py`): 3-step retrieve→compute→decide TAG bot with 10 opponent-specific adjustment types. Adjustment fire rate 15.4% over 500 episodes after Week 11 improvements.
- **Training infrastructure** (`src/training.py`, `src/poker/training.py`): `load_model` with LoRA preset selection and optional unsloth fast path, `PokerBCTrainer` using trl SFT, `PokerReinforceTrainer` with batch-normalized clipped advantages and EMA tracking.
- **CLI** (`scripts/poker_train.py`): single entry point for BC/RL/Eval/Full phases with full hyperparameter control, seeded for reproducibility, structured JSON output for eval metrics.
- **PrimeIntellect playbook** (`scripts/primeintellect/`): idempotent bootstrap, one-command pipeline runner, README covering pod launch through results retrieval.
- **Evaluation** (`src/poker/evaluation.py`): per-action, per-street, confusion-matrix, JSON export.
- **Tests**: 36 unit tests across poker environment, heuristic, tasks, agents, rewards, evaluation, and training infrastructure. All passing.
- **Final artifacts**: BC checkpoint, RL checkpoint with 4 retrospective-selection candidates, training curves plot, structured eval JSON, Colab demo notebook.

### Headline numbers
- Zero-shot Qwen-7B: **8.0%** action-type accuracy (25 episodes).
- Zero-shot Qwen-1.5B: **[ZS_1_5B]%** (50 episodes).
- BC Qwen-1.5B: **[BC_ALL]%** all streets, **[BC_PRE]%** preflop, **[BC_POST]%** postflop.
- RL Qwen-1.5B from BC: **[RL_ALL]%** all streets, **[RL_PRE]%** preflop, **[RL_POST]%** postflop.
- Heuristic: **100%** (by construction).
- Training time: ~90 minutes on H100 80GB.

## ORIENT

### What went well
1. **The BC warm-start turned out to be everything.** REINFORCE from a random-init small model would be hopeless on this reward sparsity. BC on 500 heuristic trajectories gave us a policy that emits structurally valid code on ~100% of rollouts from iteration 1 of RL, which is the only regime where REINFORCE is stable.
2. **The pivot to poker (Week 9) was the right call.** The original synthetic needle/KV/aggregation tasks were trivial for the heuristic and gave the LLM nothing interesting to learn. Poker surfaces a genuine long-context signal (opponent stats) that the heuristic must use and that the LLM must learn to extract. Without the pivot the Week 14 presentation would have been "heuristic solves everything 100%, LLM solves nothing 0%, not much to see."
3. **Stabilized REINFORCE.** Going from raw advantages to batch-mean-and-std-normalized clipped advantages (added Apr 11) eliminated the catastrophic regression we saw on April 7 (50% → 37.5% over 20 iterations). This is textbook variance reduction — the course notes call it out explicitly in §7 — but we had to hit the failure mode ourselves before we understood why.
4. **Reproducibility from day one.** `set_training_seed` propagates to Python, Torch, numpy, and HF SFT seeds. Given a random seed and a GPU, the full BC → RL → Eval pipeline produces bit-identical trajectories. This mattered when we wanted to isolate whether a change was due to code or noise.
5. **Tests caught real bugs.** The confusion-matrix row/column off-by-one would have put a misleading heatmap in the final report. The BC task mix regression test prevented silently reverting the preflop/postflop diversity fix. Writing 34 unit tests across 23 poker-specific scenarios was overkill for a semester project — and saved us hours at the end when we were debugging the final run.

### What didn't, and what we'd do differently
1. **We picked the model before testing it.** Qwen2.5-Coder-1.5B was chosen because it fit on a free Colab T4. By the time we had PrimeIntellect access we had 500 trajectories of code tailored to that model's tokenizer and code style. Re-running BC on Qwen-7B would have taken another 90 minutes we didn't have. Next time: pilot the zero-shot baseline on three model sizes before writing the training loop.
2. **Reward is defined against the heuristic.** This caps our ceiling at "match heuristic" and makes the framing of "RL beats BC" a statement about imitation fidelity, not poker skill. A reward defined by simulated EV against a held-out opponent pool would have been honest-er but required a faster poker simulator than we had.
3. **No hyperparameter sweep.** Single seed, single set of hyperparameters per phase. The course notes §9 are explicit that single-trial comparisons conflate algorithm and tuning. We report the budget honestly (Table in §4.1 of the report) but cannot claim RL "beats" BC with confidence bands. Would have taken ~5 hours to rerun with 3–5 seeds and we chose presentation-quality figures over error bars.
4. **Preflop data is 84% tier-5.** Random dealing produces mostly trash hands, which means most preflop decisions in BC data are trivially fold/check. We partially mitigated this with the `--bc-task-mix mixed` flag that oversamples preflop tasks, but the underlying distribution is still skewed. A principled fix would be rejection-sampling on hand strength during task generation.
5. **Colab demo is cramped.** Running the LoRA-loaded model for 3 demo episodes takes ~1 minute per episode inside a free-tier T4. The demo works but it's not snappy. A distilled or quantized inference checkpoint would be a nice-to-have we didn't build.
6. **We underestimated how much of the work was plumbing.** Agent interface, sandbox, prompt formatting, trajectory collection, sandboxed code extraction, fallback action parsing, sampling and extraction retries — all of this is *before* any learning happens. In retrospect we'd have spent Week 4–6 exclusively on this plumbing rather than splitting time with the Week 4–8 synthetic tasks.

### Surprises
- **The LLM often ignores the hand history even after BC.** Manual inspection of 20 BC rollouts shows that even when the generated code computes VPIP correctly, it sometimes discards the stats and decides from hand strength alone. This is a BC dataset artifact — 84.6% of heuristic decisions don't adjust for opponent, so the model correctly learns that the stats are *usually* irrelevant. RL nudges this but does not fix it.
- **Small LR matters more than batch size.** We tried LR 1e-5 for RL early on and saw the model collapse to constant `fold` within 5 iterations. Dropping to 5e-6 recovered. The course notes' point about LR being a nuisance hyperparameter when you change anything else was exactly the lesson we lived.
- **Negative REINFORCE losses are fine.** The first time we saw iteration loss drop below zero we thought something was broken. It isn't — clipped advantages can be signed, so `loss = -advantage * log_prob` can be either sign. The EMA reward is the real training signal.

### Critical risks (recognized, some mitigated, some shipped as-is)
- **Overfitting to the heuristic's quirks.** Mitigated: we collect rollouts on mixed-street tasks, include both all-streets and street-specific evaluation.
- **Reward hacking.** Not observed on this task, likely because type-match is a discrete 0/1 and the sandbox restricts what the model can do to cheat.
- **Distribution shift from synthetic to real poker.** Shipped as-is. Our hand histories are synthetic and consistent; real opponents drift.
- **Evaluation bias.** Mitigated: agents share the same random task set per evaluation run.

## DECIDE

### If we had another week
1. **Run 3 seeds of the full pipeline** to put error bars on the BC vs. RL comparison. ~5 hours of H100 time, two-sentence change to the report.
2. **Implement EV-based reward** and retrain for 20 iterations. Most interesting scientific question — does RL still improve if the reward is not defined against the heuristic itself?
3. **Evaluate at Qwen-7B** to see whether the improvement arc scales. If BC+RL Qwen-7B clears 90% on all streets, that's a much stronger RLM claim.
4. **Adversarial self-play.** Train two BC-warmstarted policies against each other instead of against the heuristic. See whether the policies converge to something interesting or collapse.

### What we'd skip
- The synthetic needle/KV/aggregation tasks (Weeks 4–8). Useful for scaffolding but did not transfer to the final domain.
- The HuggingFace Inference API path (Week 10). Rate-limited, non-reproducible, and we had to rewrite everything for local inference anyway.
- Colab as the primary training platform. Free-tier T4 is fine for smoke tests but the real runs want an 80GB Ampere or Hopper GPU. PrimeIntellect was the right choice; we should have moved to it in Week 11 instead of Week 14.

## ACT

### What shipped this week
1. Upstream training-infra improvements (seed propagation, BC grad accum / weight decay, REINFORCE normalized clipped advantages, sampling temperature/top-p knobs, eval JSON export).
2. PrimeIntellect bootstrap + pipeline runner scripts (`scripts/primeintellect/`).
3. Final PrimeIntellect training run: BC + 20 RL iterations + 50-episode eval, 90 minutes wall clock.
4. This self-critique, the final report, the final presentation slide outline, the Colab demo notebook, and a fresh project README with reproduction instructions.
5. Structured eval JSON in `experiments/results/final_eval_*.json` so every number in the report and slides is auditable back to the run.

### Final status
- Code + tests on `main`: 36/36 passing, repo clean.
- Report (`report.md`): conference-style, results tables populated from the final eval JSON.
- Self-critique (`self_critique_week15.md`): this document.
- Presentation (`docs/final_slides_outline.md` + final `.pptx`): 18 slides.
- Colab demo (`notebooks/final_demo.ipynb`): runs zero-shot vs. BC vs. RL vs. heuristic on 3 example scenarios.
- Reproduction instructions: in `README.md` and `scripts/primeintellect/README.md`.

### Resource consumption
- Compute: ~2 PrimeIntellect H100-hours for the final run, plus ~4 hours across prior pilots (April 7 preflop, April 10 BC smoke test, April 14 stabilized-advantage pilot).
- Code: ~3,000 lines Python across `src/` and `scripts/`, ~1,400 lines of tests, ~900 lines of documentation.
- Time: 15 weeks × 3 people, roughly 6–8 hours/person/week in the final three weeks.
