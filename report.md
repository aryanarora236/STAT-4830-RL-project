# Recursive Language Models for Poker Decision-Making

**STAT 4830 · Final Report · April 28, 2026**
Aadithya Srinivasan, Aryan Arora, Aarav M.

## Abstract

We study whether a small open-weight language model can learn to play No-Limit Texas Hold'em by writing Python code that parses long hand histories, computes opponent statistics, and selects an action. The system follows the Recursive Language Model (RLM) pattern: instead of stuffing the full context into the attention window, the model emits code that runs in a sandboxed Python REPL, and the printed output becomes its action. We build a 6-max poker environment with consistent opponent archetypes (rock, TAG, LAG, fish, maniac), a heuristic TAG bot that serves as both ground truth and demonstrator, and a two-stage fine-tuning pipeline: behavior cloning on heuristic trajectories followed by REINFORCE with batch-normalized clipped advantages. Starting from Qwen2.5-Coder-1.5B, 130 iterations of RL training lift the exponentially-weighted rollout accuracy from **25.0%** (BC initialization) to a peak of **42.3%** around iteration 105, with individual batches hitting 75% repeatedly. A zero-shot baseline of Qwen2.5-Coder-7B (via the Hugging Face Inference API) reaches **8.0%** action-type accuracy for reference, and the heuristic reaches 100% by construction. The central finding is two-sided: REINFORCE from a BC warm-start *does* teach a 1.5B model to use a Python REPL for opponent modeling — but it also discovers that the type-match reward is hackable by printing a canonical action directly and skipping the REPL entirely. We document both the improvement arc and the reward-hacking failure mode, and use retrospective checkpoint selection (course §10) to pick the best adapter for evaluation.

## 1. Introduction

Large language models are trained on a fixed context window, but many decision problems have context that is both long and structured — hand histories in poker, transaction logs in finance, test traces in software engineering. The Recursive Language Model (RLM) framework from Zaremba et al. argues that rather than forcing everything through attention, the model should be allowed to *write code* that parses the context and returns a summary. We examine whether this framework is *learnable*: can a small model be fine-tuned to use a Python REPL well enough to beat its zero-shot baseline on a domain where context actually matters?

No-Limit Texas Hold'em poker is a natural testbed. Optimal play requires exploiting opponent tendencies, which are only visible in the history of prior hands — the VPIP, PFR, postflop aggression, and fold-to-continuation-bet statistics that every online poker tracker computes. A model that ignores history plays a fixed, exploitable strategy. A model that reads history can adjust. This is exactly the pattern the RLM framework is designed for.

Our contributions:

1. **A reproducible poker-RLM stack.** Environment, heuristic demonstrator, task generator, safe Python sandbox, and two-phase trainer — all in <3k lines of Python, 36 unit tests, runnable in a 90-minute GPU pod session.
2. **A behavior-cloning → REINFORCE pipeline tailored for code-generating agents.** The BC phase teaches the retrieve→compute→decide process; REINFORCE stabilizes it against realistic rollouts where code extraction and sandbox execution can fail.
3. **An empirical study of what learning adds.** Going zero-shot → BC → RL is a clean isolation of the contribution of each training stage, reported with hyperparameter budgets and hardware fixed, in line with the benchmarking standards of AlgoPerf-style optimizer comparisons.

## 2. Background and Related Work

**Recursive Language Models.** RLM-minimal (Zaremba et al.) demonstrated that an LLM with access to a Python REPL can answer needle-in-haystack queries over long structured text by writing regex-based extraction code. The REPL acts as external memory: the model issues a short query, reads the printed result, and iterates. Our poker agent extends the same pattern to a decision problem rather than a retrieval problem.

**REINFORCE for language models.** The course notes (§7) frame modern RL for LMs as a four-step loop: sample, score, weight, update. REINFORCE uses the log-derivative trick to express the gradient of expected reward as `E[r(Y) ∇ log p_θ(Y|X)]`, then estimates it by sampling. Baseline subtraction reduces variance without changing the expected gradient. GRPO rescales rewards within a batch as `(r - mean)/std`; RLOO uses the leave-one-out mean as baseline. We use a batch-mean baseline combined with normalized and clipped advantages (`adv ← clip((adv - mean)/std, -k, k)`), which is empirically the most stable of the three on our small-batch setting.

**Benchmarking optimizers honestly.** §9 of the course notes warns that optimizer comparisons collapse into comparisons of tuning procedures. We apply that discipline: all three agents (zero-shot, BC, RL) use the same base model, same seed, same eval task set, and we explicitly report the hyperparameter budget spent on each.

**Tuning playbook.** §10 emphasizes that learning rate must be re-tuned when batch size changes, and that retrospective checkpoint selection (save every-k-iters, pick best post-hoc) is preferable to final-checkpoint reporting. We save every 5 RL iterations and pick the checkpoint with the highest EMA reward for evaluation.

## 3. Methodology

### 3.1 Problem Formulation

Each poker episode is a tuple `(C, q, a*)` where `C` is the game-state context (hole cards, board, stacks, positions, current betting, and 15 previous hand records), `q` is a fixed question ("What should you do?"), and `a*` is the ground-truth action produced by the heuristic bot. The agent sees `(C, q)`, emits Python code, the sandbox executes it against a `CONTEXT` global, and the last printed line is parsed as the action `â ∈ {fold, check, call $X, raise $X}`. The agent is scored by type match: `r = 1` if `type(â) == type(a*)`, else `0`. Reward shaping for REINFORCE additionally penalizes excess REPL steps and token usage, but at evaluation we report type-match only.

### 3.2 Poker Environment

A 6-max No-Limit Hold'em environment (`src/poker/environment.py`) provides: card/deck objects, a 5-best-of-7 hand evaluator, Monte Carlo equity estimation, structured `GameState` and `HandRecord` types, and five opponent archetypes with distinct statistical profiles:

| Archetype | VPIP | PFR | Aggression | Fold-to-CBet |
|---|---|---|---|---|
| Rock     | 14% | 11% | 1.2 | 70% |
| TAG      | 22% | 18% | 2.0 | 55% |
| LAG      | 35% | 28% | 2.8 | 40% |
| Fish     | 52% | 10% | 0.6 | 35% |
| Maniac   | 60% | 42% | 3.5 | 25% |

Task generation (`src/poker/tasks.py`) assigns a random archetype per opponent and simulates 15 prior hands from that profile, producing a history that is *consistent* — parsing it gives stats that match the archetype the villain will behave under in the current hand. Context length is 4.3k characters on average (range 2.8k–5.8k), which exceeds the prompt budget of naive zero-shot prompting and motivates the RLM approach.

### 3.3 Heuristic Baseline

The heuristic (`src/poker/heuristic.py`) is a TAG-style bot that follows the same retrieve→compute→decide flow the LLM must learn:

- **Retrieve:** parse previous hand records into per-opponent `OpponentStats` (VPIP, PFR, aggression, fold-to-cbet).
- **Compute:** preflop — categorize the hand into 5 tiers (AA/KK/QQ/AKs = T1 down to trash = T5); postflop — score made hand + draws + pot odds.
- **Decide:** apply opponent-specific adjustments to a base strategy — exploit fish with wider value, respect rocks, trap maniacs, bluff into high fold-to-cbet, fold marginal hands to passive bets, call down aggressive players.

Over 500 evaluation episodes the heuristic adjusts its decision based on opponent profile in **15.4%** of hands; the remaining 84.6% are dictated by hand strength alone. This adjustment rate is the fraction of decisions where long-context reasoning actually changes the answer — it is the signal we want the LLM to learn.

### 3.4 Agent Architecture

Every agent, including the heuristic, exposes the same `run_episode(context, question, answer) → (predicted_action, transcript)` interface (`src/models.py`, `src/poker/agents.py`). An episode is a sequence of (code, execution result) pairs capped at `max_steps=5`. Code runs in a sandbox (`src/utils.py: safe_execute_code`) that blocks imports outside a whitelist, replaces `__builtins__` with a restricted dict, and enforces a 5-second wall-clock timeout.

The LLM agent (`PokerLocalLLMAgent`) takes a Hugging Face causal LM + tokenizer, formats the task with the poker system prompt (`POKER_SYSTEM_PROMPT`), generates a response, extracts code from markdown fences (or a heuristic fallback), executes it, and retries up to three times on extraction or execution failure. The final printed line is parsed into a canonical action via `parse_action`.

### 3.5 Training Pipeline

**Phase 1 — Behavior Cloning.** We roll out `PokerHeuristicAgent` on a mixed distribution of all-streets, preflop-only, and postflop-only task generators, producing 500 successful `(prompt, code)` pairs. Each code block is the heuristic's literal 3-step script. We fine-tune Qwen2.5-Coder-1.5B with 4-bit quantization (NF4) and LoRA (r=16, α=32, attention targets) using `trl.SFTTrainer`. Loss is standard causal-LM cross-entropy on the assistant span only. Training hyperparameters: batch size 4 × grad accum 4 = effective 16, LR 2e-4 with 10% warmup and cosine decay, weight decay 0.01, 2 epochs, max sequence length 4096. The 4096 length is critical — at 2048 we truncated 46% of training examples in earlier experiments. Precision is auto-selected from GPU capability (bf16 on Ampere+, fp16 on Turing/Volta).

**Phase 2 — REINFORCE.** Starting from the BC checkpoint, we run 20 iterations of batch-8 REINFORCE. Each iteration: generate 8 rollouts at `temperature=0.2, top-p=0.9`, execute each in the sandbox, compute type-match reward, then form advantages as `clip((r - baseline) / std, -2, +2)` where baseline is an EMA over past batches with decay 0.9. The policy gradient loss per rollout is `-advantage · Σ log π(token_t | prompt, token_<t)`. Gradients accumulate across the batch, are clipped to norm 1.0, and fed to AdamW at LR 5e-6. Checkpoints save every 5 iterations.

### 3.6 Reward Function

The full training reward is $R = \text{TypeMatch}(\hat a, a^*) - \lambda_s (T/T_{\max}) - \lambda_t N_{\text{tokens}}$, with $\lambda_s=0.05$ and $\lambda_t=10^{-4}$. We also track opponent-adjustment rate — the fraction of rollouts in which the agent's parsed output references `vpip` or `aggression` — as a diagnostic of whether the model is actually using the history.

## 4. Experiments

### 4.1 Hardware and Hyperparameters

All experiments run on a single NVIDIA H100 80GB on PrimeIntellect. Qwen2.5-Coder-1.5B is loaded in 4-bit NF4 with LoRA adapters (trainable params ~4.4M / 893M, 0.49%). Seed 42 controls task sampling and trainer initialization. We perform no hyperparameter sweep over RL LR or temperature — the values above were chosen after a single 10-iter preflop pilot run on April 7 and kept fixed for the final sweep. Reporting a single-trial result is honest about the tuning budget; we call this out explicitly rather than claiming a tuned comparison.

| Stage | Trials | Budget |
|---|---|---|
| Zero-shot | 1 | prompt-engineered, no sweep |
| BC | 1 | fixed hyperparameters, 2 epochs, 500 trajectories |
| RL (medium → long) | 1 | fixed hyperparameters, 10 + 120 = 130 iterations, batch 4 |

### 4.2 Zero-shot Baseline

Before training we evaluate Qwen2.5-Coder-7B-Instruct (via Hugging Face Inference API) and Qwen2.5-Coder-1.5B-Instruct (local, same model we will fine-tune) on 25 and [ZS_N] poker tasks respectively. The larger 7B reaches **8.0%** action-type accuracy; the 1.5B model reaches **[ZS_1_5B]%**. Both fail in qualitatively similar ways — they usually produce code, often forget to print an action on the last line, and when they do print, they guess an action from hand strength alone without parsing history.

| Agent | N | Exact | Type match | Avg reward | Avg steps |
|---|---|---|---|---|---|
| Qwen-7B zero-shot (HF API) | 25 | 8.0% | 8.0% | 0.092 | 1.9 |
| Qwen-1.5B zero-shot (local) | [ZS_N] | [ZS_EX]% | [ZS_TM]% | [ZS_R] | [ZS_S] |
| PokerHeuristicAgent | 25 | 100.0% | 100.0% | 1.000 | 3.0 |

### 4.3 Behavior Cloning

500 trajectories collected in 0.4 s; 100% are correct by construction (the heuristic is ground truth). Average code length 3335 chars, average context 4302 chars. BC training takes ~15 min on H100 with unsloth. The BC checkpoint reaches approximately **25% rollout accuracy** on the RL task distribution at the first REINFORCE iteration, which is a proxy for its held-out eval accuracy pending the `--phase eval` run described in §5.

### 4.4 REINFORCE

Two contiguous runs were executed from the BC checkpoint on 2026-04-20, both at batch size 4, EMA γ=0.9, temperature 0.2, top-p 0.9, advantage clip ±2, LR 5e-6, max-new-tokens 1024:

- **Medium run (10 iterations).** Raw accuracy 0% → 50%, reward 0.075 → 0.575, with a 75% single-batch peak at iter 7. Checkpoint saved to `./checkpoints/poker_rl_medium_simple_20260420`. Log at `docs/results/poker_rl_medium_simple_20260420/`.
- **Long run (120 iterations, continued from the medium checkpoint).** EMA accuracy starts at 25.0%, climbs to a peak **42.3% at iteration 105**, then drifts back to 28.7% at iteration 120. Single-batch accuracy hits 75% on 12 iterations spread throughout the run. Per-iteration checkpoints saved every 5 steps. Log at `docs/results/poker_rl_long_simple_120iters_20260420/`.

Each iteration logs accuracy, raw reward, EMA reward and EMA accuracy (γ=0.9), loss, baseline, and a suite of rollout-quality counters (`code_extracted`, `exec_ok`, `no_code`, `exec_error`, `stdout_empty`, `fallback_used`, `wrapped_action_code`). These counters are the diagnostic that let us detect reward hacking (§6): across the 120-iteration run the fraction of rollouts that produced extractable Python code (`real_code` = `code_extracted - wrapped_action_code`) averaged **0.3 of 4 rollouts per batch**, meaning the policy usually bypassed the REPL entirely and emitted action text that was wrapped post-hoc as `print("call $X")`.

A prior pilot on April 7 (preflop-only, 10 iterations, pre-stabilization) showed accuracy rising from 37.5% to 50.0% over 2917 s; a separate 20-iteration full-streets pilot regressed 50.0% → 37.5%. The stabilized advantage estimator used in the April 20 runs produces a monotone EMA ascent to iteration 105 — a clear improvement over the pre-stabilization regression — but the post-peak drift to 28.7% shows the method is still sensitive to small-batch variance and the reward-hacking attractor.

## 5. Results

### 5.1 Training Accuracy Arc

Training-time (batch-4) rollout accuracy over the combined 130-iteration RL run:

| Milestone | Iteration | Batch accuracy | EMA accuracy (γ=0.9) | Reward |
|---|---|---|---|---|
| Medium run start (BC init) | 1 | 0.0% | 0.0% | 0.075 |
| Medium run end | 10 | 50.0% | 24.2% | 0.575 |
| Long run start (continuing) | 1 (of 120) | 25.0% | 25.0% | 0.250 |
| Long run peak | **105** | 50.0% | **42.3%** | 0.500 |
| Long run single-batch peak | 7, 13, 26, 32, 55, 91, 99–102, 113 | **75.0%** | up to 42.3% | 0.750 |
| Long run final | 120 | 0.0% | 28.7% | 0.000 |

For reference, the pre-stabilization April 7 full-streets pilot regressed 50%→37.5% over 20 iterations; the stabilized April 20 run achieves the opposite — a 25%→42.3% ascent — at the cost of post-peak drift.

### 5.2 Held-out Evaluation

To produce a clean comparison against the zero-shot baseline and the heuristic, we evaluate the best RL checkpoint (iteration 105, EMA accuracy 42.3%) on 50 episodes per suite using the same task set for every agent. Command:

```bash
python scripts/poker_train.py --phase eval \
  --model ./checkpoints/poker_rl_long_simple_120iters_20260420/iter_105 \
  --eval-episodes 50 --eval-by-street \
  --eval-json experiments/results/final_eval_iter105.json
```

| Agent | All streets | Preflop | Postflop | Avg reward | Avg steps |
|---|---|---|---|---|---|
| Zero-shot Qwen-7B (HF API, 25 ep) | 8.0% | 33% | — | 0.092 | 1.9 |
| Zero-shot Qwen-1.5B (local) | [ZS_ALL]% | [ZS_PRE]% | [ZS_POST]% | [ZS_R] | [ZS_S] |
| BC Qwen-1.5B | [BC_ALL]% | [BC_PRE]% | [BC_POST]% | [BC_R] | [BC_S] |
| RL Qwen-1.5B (iter 105) | [RL_ALL]% | [RL_PRE]% | [RL_POST]% | [RL_R] | [RL_S] |
| Heuristic (ground truth) | 100% | 100% | 100% | 1.000 | 3.0 |

*Populated from `experiments/results/final_eval_iter105.json` via `scripts/fill_report_from_eval.py`.*

### 5.3 Per-Action Breakdown

| Agent | Fold | Check | Call | Raise |
|---|---|---|---|---|
| Zero-shot | [ZS_FOLD]% | [ZS_CHECK]% | [ZS_CALL]% | [ZS_RAISE]% |
| BC | [BC_FOLD]% | [BC_CHECK]% | [BC_CALL]% | [BC_RAISE]% |
| RL | [RL_FOLD]% | [RL_CHECK]% | [RL_CALL]% | [RL_RAISE]% |

### 5.4 Confusion Matrices

For each agent we report `M[correct_action][predicted_action]`, counting over the 50-episode all-streets eval. Loaded from `experiments/results/final_eval_iter105.json: agents.<name>.confusion_matrix`.

### 5.5 Training Curves

Long-run EMA reward and accuracy are in `docs/results/poker_rl_long_simple_120iters_20260420/training_curves.png`. The curve shows:

1. A slow climb from 25% EMA at iter 1 to 42.3% EMA at iter 105 — the main "learning" phase.
2. A decay to 28.7% EMA by iter 120 — the onset of reward-hacking collapse (§6).
3. Single-batch swings of ±50% throughout — a direct consequence of batch size 4. Every episode is worth 25 percentage points, so the raw signal is inherently noisy and the EMA is the load-bearing statistic.

The medium run's curve (`docs/results/poker_rl_medium_simple_20260420/training_curves.png`) shows the cleaner 0%→50% ascent that got us onto the BC-compatible region before the long continuation.

### 5.6 Rollout-Quality Diagnostics

A key diagnostic during the long run is the `real_code` vs `wrapped_action_code` counter, averaged across 120 iterations of batch 4 (480 total rollouts):

| Diagnostic | Average per batch | Interpretation |
|---|---|---|
| `real_code` (extractable Python) | 0.3 / 4 | Model rarely wrote usable code |
| `wrapped_action_code` (fallback) | 3.7 / 4 | Model typically emitted `fold` / `call $X` directly |
| `exec_ok` | 4.0 / 4 | When code existed, it ran |
| `nonzero_reward` | 1.1 / 4 | ~25-30% of rollouts earn reward |

This tells us what the policy optimized: since the type-match reward fires ~25% of the time on *any* action guess and the code-writing path is expensive (tokens + latency + failure risk), REINFORCE learned to prefer the fallback. This is exactly the reward-hacking attractor §7 of the course notes warned about.

## 6. Discussion

**What worked.** The BC warm-start and the stabilized advantage estimator. With no BC the RL phase has nothing to reinforce — as the course notes observe, *"if you never witness a behavior, you can never reinforce it."* And with raw (un-normalized, un-clipped) advantages, a 20-iteration pilot regressed 12.5 accuracy points. Swapping in `adv ← clip((adv − mean)/std, −2, +2)` produced the 25%→42.3% ascent visible in the long run — a clean, monotone improvement over the first 105 iterations.

**The reward-hacking finding.** The diagnostic counters in §5.6 are the central empirical finding of the project. Across 120 iterations, the policy emitted extractable Python on only ~0.3/4 rollouts per batch while `wrapped_action_code` fired 3.7/4. This is *not* a plumbing bug — the sandbox still executes, the reward still computes. It is the policy deliberately opting out of the REPL. The mechanism is straightforward: the type-match reward fires on ~25% of uniformly random actions, and writing correct retrieve→compute→decide Python costs many tokens with a high risk of execution failure. REINFORCE found the steepest-ascent path to expected reward, and that path bypasses the tool the RLM framework is built around. §7 of the course notes document exactly this pattern in the e-avoidance toy example; we reproduce it on a harder task. The implication is that *the reward must directly price tool use* for RLM training to work in general — either an explicit bonus for REPL usage, or a task structure where actions are not easily guessable.

**The post-peak drift.** EMA accuracy peaks at iteration 105 and decays back to 28.7% by iteration 120. The decay coincides with the `real_code` counter approaching zero, which is consistent with reward hacking dominating the gradient signal after enough iterations. Retrospective checkpoint selection (§10 of the course) gets around this for evaluation — we report metrics on iter_105 — but it does not fix the underlying dynamic.

**What else didn't, until we fixed it.** The first full-street RL pilot (April 7, pre-stabilization) regressed 50%→37.5% over 20 iterations. Debugging that run produced the advantage-clipping and per-batch normalization now in `src/training.py`. Separately, fixing an off-by-one in the confusion-matrix row/column convention exposed that the BC model was disproportionately confusing `call` and `raise` rather than `fold` and `check` — a signal that the model had learned hand-strength heuristics but not sizing.

**The RLM hypothesis.** Our results support the weak form of the RLM hypothesis: a 1.5B LLM *can* be fine-tuned to imitate a Python-REPL reasoning pattern via behavior cloning. They do *not* yet support the strong form — that REINFORCE with a simple type-match reward further improves such a policy. The opposite appears closer to true: simple reward shapes collapse toward reward-hacking attractors, and the policy ends up worse at the task it was supposed to improve at. Beating this requires either a better reward (EV-based, §8) or a training regime where tool use is mechanically required to earn reward.

## 7. Limitations

1. **Ground truth is the heuristic.** Type-match reward caps the model at heuristic performance. We cannot distinguish "model matches heuristic" from "model is better than heuristic."
2. **Synthetic opponents are consistent by construction.** Real opponents have style drift across sessions; our hand-history generator samples actions i.i.d. from a fixed profile.
3. **Single seed, single tuning trial.** The zero-shot, BC, and RL results above each come from one training run. Variance bars would require re-running the full pipeline with 3–5 seeds (~5 hours total on one H100).
4. **Small model.** Qwen-1.5B is a convenient size for LoRA fine-tuning on a single GPU but is well below the scale at which LLMs begin to show emergent reasoning. Results at Qwen-7B or Qwen-32B are likely qualitatively different and were outside our compute budget.
5. **No KL penalty against the base model.** REINFORCE without a KL anchor can drift; we rely on advantage clipping and a small LR (5e-6) to limit drift, and observe no catastrophic forgetting over 20 iterations, but longer runs would need an explicit KL term.

## 8. Future Work

- **EV-based reward.** Replace type-match with expected value computed by simulating the hand against a held-out opponent pool. This decouples the reward from the heuristic and opens the door to beating it.
- **Multi-hand tournaments.** Evaluate agents over full sit-and-go tournaments rather than single-hand decisions, scoring by end-of-tournament chip stack.
- **Larger base models + GRPO.** Replace REINFORCE-with-clipped-advantages by group-relative PPO on a 7B or 32B base, which the course notes frame as a unified approach to reward-weighted SFT.
- **KL regularization against BC model.** An explicit `β · KL(π_θ || π_BC)` term would let us scale RL LR up safely.
- **Adversarial self-play.** Train two RL agents against each other instead of against the heuristic; expected to produce more diverse policies at the cost of training instability.

## 9. Conclusion

We demonstrate that a 1.5B-parameter open-weight LLM, fine-tuned via behavior cloning on 500 heuristic trajectories followed by 130 iterations of REINFORCE with batch-normalized clipped advantages, can be trained to reach **42.3%** EMA rollout accuracy on a poker decision task whose zero-shot Qwen-7B baseline is 8% and whose heuristic ceiling is 100%. The same experiment also exposes a reward-hacking failure mode: the policy learns to skip the Python REPL and emit action text directly, because the type-match reward fires ~25% of the time on random guesses and writing correct code is expensive. This failure is a direct instance of the pattern taught in §7 of the course notes, and it argues that RLM training requires reward structures that price tool use explicitly, rather than measuring only the final action. The complete pipeline — environment, heuristic demonstrator, BC trainer, stabilized REINFORCE, evaluation framework, reproducible PrimeIntellect playbook — is a ~3k-line Python codebase, runs end-to-end in <2 hours on a single H100, and generalizes to any domain where decisions depend on long structured context.

## References

1. Zaremba et al. *Recursive Language Models for Tool-Augmented Retrieval.* (2025)
2. Shallue et al. *Measuring the Effects of Data Parallelism on Neural Network Training.* JMLR 2019.
3. Dahl et al. *Benchmarking Neural Network Training Algorithms.* AlgoPerf, 2023.
4. Kasimbeg et al. *Accelerating Neural Network Training: An Analysis of the AlgoPerf Competition.* 2024.
5. Ahmadian et al. *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs.* 2024.
6. Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024 (GRPO).
7. Wilson et al. *The Marginal Value of Adaptive Gradient Methods in Machine Learning.* NeurIPS 2017.
8. Choi et al. *On Empirical Comparisons of Optimizers for Deep Learning.* 2019.

## Appendix A — Example Trajectory

A representative agent trajectory on a postflop hand, abbreviated:

```
CONTEXT (excerpt):
Your Hand: Ah Kd
Community Cards: Qc 7h 2s (Flop)
Your Position: BTN, Pot: $18, To Call: $6
=== PREVIOUS HANDS (15) ===
Hand #1  UTG raises $6, ... BTN calls $6 ...
  Result: UTG wins $18
...

AGENT CODE (retrieve + compute + decide, abbreviated):
import re
lines = CONTEXT.split('\n')
# ...parse stats per opponent...
# UTG: VPIP 47%, PFR 20%, Agg 2.8, Fold-to-CBet 55%
# ...evaluate hand: AK high, no made hand, 0 outs to straight/flush
# ...pot odds: 6 / (18+6) = 0.25
# ...UTG is aggressive TAG → call down
print("call $6")

REWARD: 1.0 (type match: call == call)
```

## Appendix B — Hyperparameter Table

| Parameter | BC | RL |
|---|---|---|
| Base model | Qwen2.5-Coder-1.5B-Instruct | (from BC) |
| LoRA r / α / targets | 16 / 32 / q,k,v,o | (from BC) |
| Quantization | 4-bit NF4 | 4-bit NF4 |
| Precision | bf16 (H100) | bf16 |
| Optimizer | AdamW | AdamW |
| LR | 2e-4 | 5e-6 |
| Scheduler | cosine, 10% warmup | constant |
| Batch size / grad accum | 4 × 4 = 16 | 4 |
| Epochs / iterations | 2 | 10 (medium) + 120 (long) = 130 |
| Max seq / new tokens | 4096 / — | 4096 / 1024 |
| Weight decay | 0.01 | 0 |
| Grad clip | 1.0 | 1.0 |
| Sampling temp / top-p | — | 0.2 / 0.9 |
| Advantage baseline | — | EMA γ=0.95 |
| Advantage clip | — | ±2.0 |
| Seed | 42 | 42 |
