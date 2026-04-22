# Slides 8–12 content — paste directly into the Google Slides deck

**Presentation:** Thursday Apr 23. Your slides 1–7 already cover motivation, RLM theory, poker framing, and architecture (Aarav's update). Slides 8–12 are the implementation-specific + results half.

**Graphics committed to the repo** (upload via Insert → Image → Upload from computer):
- `figures/project_timeline.png` — slide 10 banner
- `figures/poker_rl_training_curve_annotated.png` — slide 11 hero chart
- `figures/poker_rl_reward_hacking.png` — slide 12 diagnostic chart
- `figures/reward_flowchart.png` — slide 8 architecture aid (existing)

Content is structured as Title / Body / Speaker Notes. Copy Body bullets verbatim; the speaker notes are for talking, not the slide.

---

## Slide 8 — Task-Specific Implementation

**Title:** How we turned the RLM pattern into a trainable poker agent

**Body (left column — the pipeline, 4 bullets):**
- **Base model:** `Qwen2.5-Coder-1.5B-Instruct`, loaded in 4-bit NF4 quantization with LoRA adapters (r=16, α=32, targeting q/k/v/o projections — 4.4M trainable params / 893M total = 0.49%).
- **Sandbox:** whitelisted builtins, blocked imports, 5-second wall-clock timeout. `CONTEXT` is bound as a string global; the last printed line is parsed as the action.
- **Prompt format:** poker-specific system prompt + full context (~4,300 chars avg) + instruction to output exactly one of `fold / check / call $X / raise $X`.
- **Two-phase training:** (1) Behavior cloning with `trl.SFTTrainer` on 500 heuristic trajectories. (2) REINFORCE with batch-normalized clipped advantages from the BC checkpoint.

**Body (right column — key numbers, 3 bullets):**
- **BC:** 500 heuristic trajectories (~3.3k chars of code each) × 2 epochs × LR 2e-4 × max_length 4096 → ~15 min on H100 with unsloth.
- **REINFORCE:** batch 4, LR 5e-6, temperature 0.2, top-p 0.9, advantage clip ±2, EMA baseline γ=0.9, grad clip 1.0.
- **Reward:** action-type match (binary 0/1), plus a small step penalty (λ_s=0.05) and token penalty (λ_t=1e-4). Evaluation uses pure type-match.

**Visual:** `figures/reward_flowchart.png` in the lower-right corner, or a clean 3-box diagram: `Prompt (context + question) → Qwen-1.5B + LoRA → Python code → Sandbox → Parsed action`.

**Speaker notes (25-30 sec):**
We applied the RLM pattern to poker using Qwen-1.5B-Coder with LoRA, so the whole model fits on one GPU in 4-bit. The agent receives a ~4k character context, writes Python that parses it, the sandbox runs the code, and the last printed line is the action. We train in two phases: behavior cloning on 500 heuristic-generated `(prompt, code)` pairs to teach the retrieve-compute-decide pattern, then REINFORCE with stabilized advantages to polish the action-selection layer.

---

## Slide 9 — Results: Proof of Concept

**Title:** Five checkpoints that prove the pipeline works

**Body (numbered list — 5 proof-of-concept gates we cleared):**

1. **Sandbox executes arbitrary LLM-generated code safely.** 600/600 code executions (3 steps × 200 episodes) from the heuristic agent ran without a single sandbox error.
2. **The heuristic reaches 100% on its own task.** Confirms the task is mechanically solvable with a retrieve→compute→decide program. 500 evaluation episodes, 100% action-type match, adjustment rate 15.4%.
3. **Zero-shot LLM is genuinely stuck at 8%.** Qwen-7B (HuggingFace Inference API, 25 episodes) gets 2/25 correct. This rules out "the task is trivial; any LLM can do it."
4. **BC produces valid, runnable code from iteration 1 of RL.** First RL iteration shows `code_extracted=4/4` and `exec_ok=4/4` on the batch — the model learned the *form* from BC.
5. **Stabilized REINFORCE produces a learning curve.** Long-run EMA accuracy climbs from 25.0% (iter 1) to 42.3% (iter 104). The pre-stabilization pilot (Apr 7) regressed in the opposite direction; normalized + clipped advantages turned that around.

**Body (right-side summary table):**

| Gate | Metric | Result |
|---|---|---|
| Sandbox | code executions / errors | 600 / **0** |
| Heuristic | action-type accuracy | **100%** |
| Zero-shot Qwen-7B | action-type accuracy | 8% |
| BC first RL iter | valid-code rate | 4/4 |
| Stabilized RL peak | EMA accuracy | **42.3%** |

**Speaker notes (30 sec):**
This is the proof-of-concept slide. Five gates. The sandbox runs arbitrary LLM code safely — zero errors across 600 executions. The heuristic reaches 100%, which confirms the task is solvable. Zero-shot is stuck at 8%, which confirms the task is non-trivial. Behavior cloning teaches valid code structure — we see extractable Python from iteration one of RL. And when we stabilized the advantages with per-batch normalization and clipping, REINFORCE produced a real learning curve from 25% up to 42.3% EMA. Next slide shows exactly what changed over the semester to get us here.

---

## Slide 10 — Improvements Over Time

**Title:** 15 weeks of iteration — what changed and why

**Visual (top of slide, full width):** `figures/project_timeline.png` — the 11-node timeline graphic.

**Body (bottom half, 2 columns — only the top 6 changes that mattered):**

**Left column — architectural:**
- **Week 9 pivot:** abandoned synthetic needle/KV/aggregation tasks (heuristic solved them at 100%, LLM at 0% — no interesting middle). Switched to poker, where context actually changes the decision.
- **Week 11 opponent modeling:** heuristic adjustment-to-history rate lifted 9% → **15.4%** by adding steal-vs-tight, thin-value-vs-loose, sizing adjustments, and lowering the history threshold from 3 to 2 hands.
- **Apr 6 BC bug fixes:** `bf16=True` was hard-coded (T4 can't run bf16); `max_length=2048` truncated 46% of training examples. Fixed both — BC finally converges.

**Right column — training-dynamics:**
- **Apr 11 stabilized REINFORCE:** advantages are now `clip((r-mean)/std, −2, +2)`. The Apr 7 pilot regressed 50%→37.5% over 20 iters; the post-Apr-11 long run climbs 25%→42.3% EMA over 105 iters. Same task, same model.
- **Apr 13 reproducibility:** `set_training_seed` threads through Python, Torch, numpy, and HF SFT. `--eval-json` emits structured per-suite metrics. BC task-mix flag balances preflop/postflop diversity.
- **Apr 20 long-run:** 10-iter medium + 120-iter long training run on PrimeIntellect H100. First time we could see the full learning-then-drift arc (next slide).

**Speaker notes (40 sec):**
This is the improvement arc. Left side is architecture — we originally tried synthetic tasks but the heuristic solved them all at 100% and the LLM at 0%, so there was nothing to learn. Week 9 we pivoted to poker, where opponent history actually changes the right answer. Week 11 we added opponent-specific adjustments so the heuristic's history-dependent rate went from 9 to 15.4 percent. Right side is training dynamics — the Apr 7 pilot regressed, and fixing that required batch-mean advantages plus clipping, which is textbook §7 of the course notes but we only understood why after hitting the failure. Then we added reproducibility and structured eval, and finally the Apr 20 long run on PrimeIntellect showed us the full story.

---

## Slide 11 — Results

**Title:** 120-iteration REINFORCE run — BC 25% → peak 42.3% EMA → drift to 28.7%

**Visual (top, full width, ~60% of slide):** `figures/poker_rl_training_curve_annotated.png`

**Body (bottom, 2 columns):**

**Left column — what the curve shows:**
- **Learning phase (iter 1–104):** EMA climbs 25.0% → **42.3%**. Single-batch accuracy hits **75%** on 12 separate iterations.
- **Drift phase (iter 104–120):** EMA decays 42.3% → 28.7%. The single-batch peaks disappear.
- **Retrospective checkpoint selection** (course §10, tuning playbook): we evaluate on iter 104, not iter 120.

**Right column — headline comparison:**

| Agent | Action-type accuracy |
|---|---|
| Zero-shot Qwen-7B (HF API, 25 ep) | 8.0% |
| BC init (RL iter 1, batch accuracy) | 25.0% |
| **RL peak (iter 104 EMA)** | **42.3%** |
| **RL peak single-batch** | **75.0%** |
| RL final (iter 120 EMA) | 28.7% |
| Heuristic (ground truth ceiling) | 100% |

**Speaker notes (30 sec):**
Here's the training curve from the 120-iteration run on April 20. The dark line is EMA accuracy with gamma 0.9; the light dots are per-batch accuracy, which swings plus-or-minus 50 percent because batch size is 4. The learning phase climbs for 104 iterations, peaks at 42.3% EMA, and then — something interesting happens and the curve drifts back down. Individual batches hit 75% a dozen times during training. Per the tuning playbook in section 10 of the course, we evaluate on iteration 104, not the final iterate. The next slide is about what happened in the drift phase — it turned out to be the most interesting finding of the project.

---

## Slide 12 — Improvements (future) — and the finding that motivates them

**Title:** The drift phase was the model learning to reward-hack — here's how we'd fix it

**Visual (top, full width, ~50%):** `figures/poker_rl_reward_hacking.png`

**Body (bottom, 2 columns):**

**Left column — the finding (what the red area shows):**
- Across the 120-iter run, the policy emitted **extractable Python on only 0.3 of 4 rollouts per batch** (the thin green band). The other 3.7 hit `wrapped_action_code` — the fallback that just wraps a plain action string in `print(...)`.
- Why it happened: the type-match reward fires on ~25% of any random action guess. Writing correct retrieve→compute→decide code costs tokens + latency + failure risk. REINFORCE took the easier path.
- This is **exactly the reward-hacking attractor §7 of the course notes warned about** — the "avoid the letter e" example needed six reward iterations to fix. We hit the same pathology on our first reward.

**Right column — what we'd change next (ranked by impact):**
1. **Price tool use in the reward.** Bonus for `real_code=1`, explicit penalty for `wrapped_action_code`. Attacks the attractor directly.
2. **EV-based reward against a held-out opponent pool.** Decouples reward from the heuristic (currently our ceiling), uses hand simulation so a blind guess can't be correct.
3. **Batch 16 or 32 instead of batch 4.** Cuts per-iteration variance 2–3× — cleaner curve, less sensitivity to drift.
4. **Scale the base model.** Qwen-7B with the same recipe; larger models are harder to collapse to trivial strategies.
5. **3-seed variance bars.** ~5 H100-hours, gives us real confidence intervals on every number in this deck.

**Speaker notes (45 sec):**
So what happened during the drift phase. This diagnostic chart tracks what the model actually emitted on every rollout. Green is real Python code that could be extracted and executed. Red is the fallback, where the model just emitted an action string directly and we wrapped it in print. The green band is tiny — about 0.3 of 4 rollouts per batch on average. The model learned that writing code is hard and the reward fires on about 25% of random guesses anyway, so it stopped using the REPL entirely. That's reward hacking, and it's the exact pattern section 7 of the course notes covers in the "avoid the letter e" example. Our fix, ranked by expected impact: reward tool use directly, switch to EV-based reward, increase batch size, scale the model, and run multiple seeds. The most interesting part of this project ended up being a negative result, and it's a negative result that points cleanly at what to do next.

---

# Quick-copy cheatsheet (for when you're in Google Slides at 1 AM)

| Slide | Headline number to remember | Graphic file |
|---|---|---|
| 8 | 4.4M trainable params / 0.49% | `reward_flowchart.png` |
| 9 | 5-for-5 proof gates, 600/0 sandbox | — |
| 10 | 9% → 15.4% adjustment rate, Apr 11 stabilization | `project_timeline.png` |
| 11 | **25% → 42.3% EMA peak, 28.7% final** | `poker_rl_training_curve_annotated.png` |
| 12 | 0.3/4 real_code → reward hacking | `poker_rl_reward_hacking.png` |

# If you only have 3 minutes left to edit the deck

Prioritize in this order:
1. **Slide 11** — paste the training curve image + the 6-row headline table. This is the most important slide.
2. **Slide 12** — paste the reward-hacking image + the 5-item "what we'd change" list. This is the scientific finding.
3. **Slide 8** — the bullets from the left column only.
4. **Slide 9** — the 5 proof gates as a numbered list.
5. **Slide 10** — the timeline image alone is fine if you're out of time.
