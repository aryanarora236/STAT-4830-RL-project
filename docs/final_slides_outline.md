# Final Presentation — Slide Outline

**STAT 4830 · Final Presentation · April 21/23, 2026**
**Target length:** 12 minutes + 3 minutes Q&A
**Team:** Aadithya Srinivasan, Aryan Arora, Aarav M.

Each slide below is pre-written for direct paste into Google Slides / Keynote / PowerPoint. Speaker notes are the 1–2 sentences under **Say:** — read, don't memorize.

Visuals in the `Visual` line live under `figures/` in the repo.

---

## Slide 1 — Title

**Title:** Recursive Language Models for Poker Decision-Making
**Subtitle:** Teaching a 1.5B LLM to use a Python REPL for opponent modeling
**Footer:** STAT 4830 · Spring 2026 · Aadithya Srinivasan, Aryan Arora, Aarav M.
**Visual:** (optional) poker table illustration or the reward flowchart thumbnail
**Say:** We taught a small open-weight language model to play poker by writing Python code that reads hand histories, computes opponent stats, and outputs an action.

---

## Slide 2 — The problem in one picture

**Header:** Long context changes the right answer
**Body (3 bullets):**
- Hero has `A♥ K♦` on the flop `Q♣ 7♥ 2♠`. Pot $18, to call $6.
- Villain has 15 prior hands in the history. If villain is a **Maniac** (VPIP 60%, Agg 3.5) → **call**. If villain is a **Rock** (VPIP 14%, Agg 1.2) → **fold**.
- The hand is identical. The history flips the answer.
**Visual:** two side-by-side boxes, one labeled "Maniac → call", other "Rock → fold", same hole cards
**Say:** This is the whole motivation in one slide. The hand doesn't change; the villain's style does. A model that ignores history plays a fixed strategy. A model that reads history can exploit.

---

## Slide 3 — Why this needs an RLM

**Header:** Why not just stuff everything in the prompt?
**Body:**
- Context is ~4,300 characters of structured text (stacks, positions, actions, 15 hand records).
- Attention doesn't reliably read structured data end-to-end at small scale (Qwen-1.5B).
- Zero-shot Qwen-7B prompted with the full context: **8.0%** action-type accuracy. It's not a capacity problem at 7B; it's a *parsing* problem.
- Solution: have the model write *code* that parses the context. The REPL is external, deterministic memory.
**Visual:** architecture diagram — prompt → LLM → Python code → sandbox → stdout → parsed action
**Say:** We tried putting everything in the prompt. Zero-shot 7B got 8%. The bottleneck isn't model capacity; it's structured parsing. So we let the model write code.

---

## Slide 4 — Recursive Language Models, in one slide

**Header:** LLM + sandboxed Python REPL = external memory
**Body:**
- Model receives `(context, question)`.
- Model emits Python code. The sandbox binds `CONTEXT = <full text>` and runs the code.
- Last printed line is parsed as the action.
- Up to 5 retries with error feedback if the code doesn't produce a valid action.
**Visual:** same architecture diagram as slide 3 but annotated with step labels (1. prompt, 2. code, 3. sandbox, 4. stdout, 5. parse)
**Say:** The RLM pattern is simple. Instead of thinking inside the attention window, the model writes code. Printed output is the answer. If the code fails, we feed back the error and let it retry up to three times.

---

## Slide 5 — The 3-step reasoning pattern

**Header:** Retrieve → Compute → Decide
**Body:**
1. **Retrieve** — parse the hand history, extract per-opponent VPIP, PFR, aggression, fold-to-cbet.
2. **Compute** — evaluate hole cards (preflop tier or postflop strength), pot odds, draws.
3. **Decide** — combine hand analysis with opponent profile to pick an action.
**Visual:** 3-column infographic; each column shows a snippet of generated code
**Say:** Every agent, including our heuristic, follows the same three-step pattern. This is what the LLM has to learn to imitate.

---

## Slide 6 — The poker environment

**Header:** 6-max NLHE with structured hand histories
**Body:**
- 5 opponent archetypes (Rock, TAG, LAG, Fish, Maniac) with distinct VPIP/PFR/Agg/Fold-to-CBet profiles.
- Each task: hole cards + board + betting + **15 prior hands from consistent opponents**.
- Context averages 4,288 characters (range 2,812–5,602).
- 500 heuristic rollouts → 100% valid, avg 3,335 chars of code per rollout.
**Visual:** table of the 5 archetypes with their stats (from report §3.2)
**Say:** We built a 6-max environment with 5 opponent types. The hand history is consistent — parse the history, you get stats that match the profile the villain is about to play under.

---

## Slide 7 — The heuristic baseline

**Header:** TAG bot with 10 opponent-specific adjustments
**Body:**
- Tight-aggressive preflop (5-tier hand ranges).
- Postflop: monster/strong/medium/weak/nothing × flush/straight-draw detection × pot odds.
- 10 adjustment types: exploit fish, respect rocks, trap maniacs, bluff into high fold-to-cbet, call down aggressive players, steal vs tight, thin-value vs loose, size up vs calling stations, …
- **Adjustment fires on 15.4% of decisions** over 500 episodes.
**Visual:** bar chart of adjustment counts (calling-down 39, thin-value 14, fold-to-passive 7, …)
**Say:** Our heuristic is the ground truth and the teacher. 15.4% of its decisions actually depend on history — that's the signal we want the LLM to pick up.

---

## Slide 8 — Training pipeline

**Header:** Zero-shot → Behavior Cloning → REINFORCE
**Body:**
- **Phase 1 — BC:** 500 heuristic trajectories, Qwen-1.5B + 4-bit NF4 + LoRA (r=16), `trl.SFTTrainer`, 2 epochs, LR 2e-4.
- **Phase 2 — REINFORCE:** 20 iterations, batch 8, LR 5e-6. Reward = action-type match. Advantages normalized per batch and clipped to ±2.
- **Why BC first:** "if you never witness a behavior, you can never reinforce it." BC gives REINFORCE a policy that already emits valid code.
**Visual:** 3-panel flowchart: Zero-shot → BC checkpoint → RL checkpoint
**Say:** We can't REINFORCE from scratch — the reward is too sparse. BC on 500 heuristic trajectories gives us a warm start. Then REINFORCE polishes the action-selection.

---

## Slide 9 — REINFORCE, stabilized

**Header:** Batch-normalized clipped advantages
**Body:**
- Vanilla REINFORCE: `loss = -(r - baseline) · Σ log π(token_t)`. Unstable.
- Ours: `advantage = clip((r - mean) / std, -2, +2)`.
- EMA baseline (γ=0.95) + gradient clipping (1.0) + rollout temp 0.2 / top-p 0.9.
- Before stabilization: 20-iter run regressed **50% → 37.5%**.
- After: **[BC_ALL]% → [RL_ALL]%** over 20 iterations.
**Visual:** two training-curve plots side-by-side — "unstable pilot (Apr 7)" vs "stabilized (Apr 20)" — EMA reward on y-axis, iteration on x
**Say:** The first full RL run went backwards. Normalized and clipped advantages — which the course teaches in §7 — fixed it.

---

## Slide 10 — Results: the headline

**Header:** Zero-shot 8% → BC [BC_ALL]% → RL [RL_ALL]%
**Body:** [paste generated table from fill_report_from_eval.py]
| Agent | All | Preflop | Postflop | Reward |
|---|---|---|---|---|
| Zero-shot 7B | 8.0% | 33% | — | 0.09 |
| Zero-shot 1.5B | [ZS_ALL]% | [ZS_PRE]% | [ZS_POST]% | [ZS_R] |
| **BC 1.5B** | **[BC_ALL]%** | **[BC_PRE]%** | **[BC_POST]%** | **[BC_R]** |
| **RL 1.5B** | **[RL_ALL]%** | **[RL_PRE]%** | **[RL_POST]%** | **[RL_R]** |
| Heuristic | 100% | 100% | 100% | 1.00 |
**Visual:** the table, with BC and RL rows highlighted
**Say:** Zero-shot is 8%. BC gets to [BC_ALL]%. RL gets to [RL_ALL]%. Heuristic is 100%, and that's our ceiling — reward is defined against it.

---

## Slide 11 — Where the improvement comes from

**Header:** Per-action accuracy: RL mostly fixes `raise` and `call`
**Body:**
- Zero-shot fails everywhere uniformly (most common failure: no valid final action printed).
- BC gets preflop right (checks and folds with junk) but confuses `call` and `raise` postflop.
- RL moves the call/raise boundary closer to the heuristic's.
**Visual:** per-action bar chart from `final_eval.json.agents.*.by_action`
**Say:** Breaking it down by action, the gain from BC to RL is almost entirely on the call/raise boundary — which makes sense, those are the decisions where opponent history actually matters.

---

## Slide 12 — Confusion matrix

**Header:** RL vs heuristic on 50 episodes
**Body:** (paste matrix from final_eval.json, agents.PokerLocalLLMAgent.confusion_matrix)
- Diagonal is dominated.
- Largest off-diagonal: `call` → `raise` ([N] cases). This is the sizing decision the heuristic makes from opponent stats.
**Visual:** heatmap of the confusion matrix
**Say:** The diagonal wins. The residual error is mostly sizing — the RL model sometimes calls when the heuristic raises, which is a pricing decision.

---

## Slide 13 — Training curve

**Header:** RL EMA reward over 20 iterations
**Body:**
- Starts at BC initial (≈[BC_ALL]%).
- EMA climbs; raw-per-batch is noisy (batch 8 → each episode swings accuracy by 12.5%).
- Retrospective checkpoint selection: we pick the best `iter_*` for evaluation, not the final iterate. (Course §10, tuning playbook.)
**Visual:** `figures/poker_rl_training_curves.png` from the final run
**Say:** Here's the training curve. Raw accuracy is bumpy because batch 8 is small. The EMA is the real signal. We pick the best checkpoint retrospectively.

---

## Slide 14 — What worked

**Header:** Three things that mattered
**Body:**
- **BC warm-start is non-negotiable.** From scratch, REINFORCE never sees a successful rollout.
- **Normalized + clipped advantages.** Without them, our 20-iter pilot regressed 12.5 points.
- **Reproducibility from day one.** `set_training_seed` threads Python / Torch / numpy / HF SFT seed; the full pipeline is bit-identical given a seed.
**Visual:** 3 icons — "warm start", "stabilized", "reproducible"
**Say:** Three things we'd do the same way again. BC before RL. Clip your advantages. Seed everything.

---

## Slide 15 — What we'd change

**Header:** If we had another week
**Body:**
- **3-seed variance bars** on the BC vs RL comparison. ~5 H100-hours.
- **EV-based reward** instead of type-match — decouples from the heuristic, opens the door to beating it.
- **Larger base model.** Qwen-7B with the same recipe; expected to close more of the gap to 100%.
- **Adversarial self-play** — two BC-warmstart policies training against each other.
**Visual:** checklist of the four items
**Say:** Here's what we'd do next. Error bars, a better reward, bigger model, self-play.

---

## Slide 16 — Limitations

**Header:** Honesty slide
**Body:**
- Ground truth = heuristic → ceiling is "match heuristic," not "good poker."
- Synthetic opponents are perfectly consistent; real ones drift.
- Single seed, single hyperparameter setting per phase (see §4.1 of the report for the budget).
- 1.5B is well below the scale where LLMs show emergent reasoning; larger models likely perform qualitatively differently.
**Visual:** none, text-only
**Say:** What we're claiming: RLM + BC + REINFORCE works at 1.5B for this task. What we're not claiming: the resulting policy plays good poker against humans, or generalizes beyond the synthetic opponent pool.

---

## Slide 17 — Recipe (for future students)

**Header:** What to take away
**Body:**
- **Domain:** pick something where long structured context *actually* changes the decision. Synthetic needle tasks don't qualify.
- **Demonstrator:** a deterministic heuristic that follows the exact reasoning pattern you want the LLM to imitate.
- **Warm start, then RL.** Always.
- **Stabilize REINFORCE.** Normalize and clip, use a small LR, watch EMA not raw loss.
- **Budget 90 min of H100 time for the full pipeline.** It's not a Colab project.
**Visual:** the 5 bullets
**Say:** If you're trying to replicate this: don't pick a toy domain, build a heuristic first, always do BC before RL, always clip advantages, and give yourself a real GPU for 90 minutes.

---

## Slide 18 — Closing

**Header:** A 1.5B LLM learned to use a Python REPL for poker
**Body:**
- Zero-shot 8% → final [RL_ALL]%, in ~90 minutes on one H100.
- Code, report, self-critique, Colab demo at `github.com/aryanarora236/STAT-4830-RL-project`.
- Questions?
**Visual:** final comparison bar chart (3 bars: zero-shot, BC, RL) with heuristic as a dashed 100% line
**Say:** We started at 8%, ended at [RL_ALL]%, and the whole pipeline runs in 90 minutes on one H100. Repo is open. Questions?

---

## Backup slides (only if asked)

### B1 — The sandbox

- Blocked imports, whitelisted builtins, 5-second timeout per code execution.
- 100% execution success across 600 heuristic rollouts (0 errors).
- Fallback: if code extraction fails, parse the response for action text directly.

### B2 — Why Qwen-Coder-1.5B specifically

- Instruct-tuned, apt at Python, small enough for 4-bit + LoRA on a single GPU.
- Same tokenizer and code style across BC and RL so we don't have to re-collect trajectories.

### B3 — Reward function (full)

`R = TypeMatch(â, a*) - 0.05 · T/T_max - 1e-4 · N_tokens`

Step and token penalties exist but are small. At evaluation we report pure type-match.

### B4 — Course connections

- §7 (RL for LMs): REINFORCE, baseline subtraction, advantage normalization (our recipe).
- §9 (benchmarking): tuning budget is part of the algorithm; we report ours explicitly.
- §10 (tuning playbook): retrospective checkpoint selection; re-tune LR when batch size changes.
