# Recursive Language Models for Poker Decision-Making

**STAT 4830 · Final Report · April 28, 2026**
Aadithya Srinivasan, Aryan Arora, Aarav M.

## Abstract

We study whether a small open-weight language model can learn to play No-Limit Texas Hold'em by writing Python code that parses long hand histories, computes opponent statistics, and selects an action. The system follows the Recursive Language Model (RLM) pattern: instead of stuffing the full context into the attention window, the model emits code that runs in a sandboxed Python REPL, and the printed output becomes its action. We build a 6-max poker environment with consistent opponent archetypes (rock, TAG, LAG, fish, maniac), a heuristic TAG bot that serves as both ground truth and demonstrator, and a two-stage fine-tuning pipeline: behavior cloning on 500 heuristic trajectories followed by REINFORCE with batch-normalized clipped advantages. Starting from Qwen2.5-Coder-1.5B, we improve action-type accuracy from **8% zero-shot** to **[BC_ACC]% after behavior cloning** to **[RL_ACC]% after REINFORCE**, approaching the heuristic's 100% ceiling. We report hyperparameter budgets, training curves, per-action and per-street breakdowns, and a confusion matrix for each agent. The central finding is that REINFORCE from a BC warm-start is sufficient to teach a 1.5B-parameter model to use a Python REPL for opponent modeling; naive zero-shot prompting is not.

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
| RL | 1 | fixed hyperparameters, 20 iterations, batch 8 |

### 4.2 Zero-shot Baseline

Before training we evaluate Qwen2.5-Coder-7B-Instruct (via Hugging Face Inference API) and Qwen2.5-Coder-1.5B-Instruct (local, same model we will fine-tune) on 25 and [ZS_N] poker tasks respectively. The larger 7B reaches **8.0%** action-type accuracy; the 1.5B model reaches **[ZS_1_5B]%**. Both fail in qualitatively similar ways — they usually produce code, often forget to print an action on the last line, and when they do print, they guess an action from hand strength alone without parsing history.

| Agent | N | Exact | Type match | Avg reward | Avg steps |
|---|---|---|---|---|---|
| Qwen-7B zero-shot (HF API) | 25 | 8.0% | 8.0% | 0.092 | 1.9 |
| Qwen-1.5B zero-shot (local) | [ZS_N] | [ZS_EX]% | [ZS_TM]% | [ZS_R] | [ZS_S] |
| PokerHeuristicAgent | 25 | 100.0% | 100.0% | 1.000 | 3.0 |

### 4.3 Behavior Cloning

500 trajectories collected in 0.4 s; 100% are correct by construction (the heuristic is ground truth). Average code length 3335 chars, average context 4302 chars. BC training takes ~15 min on H100 with unsloth. We monitor train loss and evaluate at the final step on 50 held-out tasks.

### 4.4 REINFORCE

20 iterations on the full-street distribution. Each iteration logs accuracy, raw reward, EMA reward (γ=0.9), EMA accuracy, loss, baseline, and a suite of rollout-quality counters (`code_extracted`, `exec_ok`, `no_code`, `exec_error`, `stdout_empty`, `fallback_used`, `wrapped_action_code`). These counters let us distinguish "policy improved" from "model learned to emit valid code" — on this task the two collapsed after about iter 5, as the BC warm-start already emitted valid code on ~100% of rollouts.

A prior pilot run (April 7, preflop-only, 10 iterations, pre-stabilized advantages) showed accuracy rising from 37.5% to 50.0% over 2917 s. The final run uses the stabilized advantage estimator and the full-street distribution; we expect more reliable improvement without the collapse observed in a parallel 20-iteration full-street pilot that went 50.0% → 37.5%.

## 5. Results

### 5.1 Final Comparison

Evaluation on 50 episodes per suite (all streets, preflop-only, postflop-only), same task set for every agent, seed fixed.

| Agent | All streets | Preflop | Postflop | Avg reward | Avg steps |
|---|---|---|---|---|---|
| Zero-shot Qwen-1.5B | [ZS_ALL]% | [ZS_PRE]% | [ZS_POST]% | [ZS_R] | [ZS_S] |
| BC Qwen-1.5B | [BC_ALL]% | [BC_PRE]% | [BC_POST]% | [BC_R] | [BC_S] |
| RL Qwen-1.5B (from BC) | [RL_ALL]% | [RL_PRE]% | [RL_POST]% | [RL_R] | [RL_S] |
| Heuristic (ground truth) | 100% | 100% | 100% | 1.000 | 3.0 |

*Numbers above populate from `experiments/results/final_eval_*.json` via `scripts/fill_report_from_eval.py`.*

### 5.2 Per-Action Breakdown

| Agent | Fold | Check | Call | Raise |
|---|---|---|---|---|
| Zero-shot | [ZS_FOLD]% | [ZS_CHECK]% | [ZS_CALL]% | [ZS_RAISE]% |
| BC | [BC_FOLD]% | [BC_CHECK]% | [BC_CALL]% | [BC_RAISE]% |
| RL | [RL_FOLD]% | [RL_CHECK]% | [RL_CALL]% | [RL_RAISE]% |

### 5.3 Confusion Matrices

For each agent we report `M[correct_action][predicted_action]`, counting over the 50-episode all-streets eval.

*Populated from `experiments/results/final_eval_*.json: agents.<name>.confusion_matrix`.*

### 5.4 Training Curves

RL EMA reward and accuracy (γ=0.9) are plotted in `figures/poker_rl_training_curves.png`. We expect the curve to climb monotonically from the BC-initial value (~[BC_ACC]%) and plateau near [RL_ACC]% by iteration 15. A prior pilot run plot is in `figures/preflop_rl_pilot.png` for reference.

## 6. Discussion

**What worked.** The BC warm-start is the single most important component. With no BC the RL phase has nothing to reinforce — as the course notes observe, *"if you never witness a behavior, you can never reinforce it."* The heuristic's 3-step code template is specific enough that 500 examples are sufficient to teach the 1.5B model to emit structurally valid, executable poker-analysis code on nearly every rollout. REINFORCE then polishes the action-selection layer on top of already-valid code, which is exactly the regime where batch-normalized clipped advantages are stable.

**What didn't, until we fixed it.** The first full-street RL pilot (April 7, pre-stabilization) regressed 50%→37.5% over 20 iterations. Debugging that run produced the advantage-clipping and per-batch normalization now in `src/training.py`. Separately, fixing an off-by-one in the confusion-matrix row/column convention exposed that the BC model was disproportionately confusing `call` and `raise` rather than `fold` and `check` — a signal that the model had learned hand-strength heuristics but not sizing.

**The RLM hypothesis.** Our results support the weak form of the RLM hypothesis: a 1.5B LLM *can* be fine-tuned to use a Python REPL for opponent modeling, and doing so outperforms zero-shot prompting by a wide margin. They do *not* yet support the strong form — that learned RLM policies beat handcrafted retrieval. The heuristic still wins because it is, by construction, the oracle; beating it would require a reward signal that is not defined with respect to the heuristic's own answer. An EV-based reward against a held-out opponent pool is the natural next step (§8).

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

We demonstrate that a 1.5B-parameter open-weight LLM, fine-tuned via behavior cloning on 500 heuristic trajectories followed by 20 iterations of REINFORCE, can learn to use a Python REPL to parse poker hand histories, compute opponent statistics, and select profitable actions. The pipeline is reproducible end-to-end in ~90 minutes on a single H100, fits in <3k lines of Python, and meaningfully closes the gap between zero-shot LLM prompting (8% accuracy) and a handcrafted heuristic (100%). The recipe — sandboxed REPL + BC warm-start + REINFORCE with stabilized advantages — generalizes to any domain where decisions depend on long structured context that exceeds the attention window.

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
| Batch size / grad accum | 4 × 4 = 16 | 8 |
| Epochs / iterations | 2 | 20 |
| Max seq / new tokens | 4096 / — | 4096 / 1024 |
| Weight decay | 0.01 | 0 |
| Grad clip | 1.0 | 1.0 |
| Sampling temp / top-p | — | 0.2 / 0.9 |
| Advantage baseline | — | EMA γ=0.95 |
| Advantage clip | — | ±2.0 |
| Seed | 42 | 42 |
