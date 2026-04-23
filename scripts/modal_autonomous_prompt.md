# Autonomous Modal prompt — shaped-reward poker RL experiment

Paste the contents of the fenced block at the bottom of this file into Claude Code. It briefs a fresh session to execute the full experiment end-to-end on Modal without further intervention from you.

---

## What this prompt does, in plain English

- Writes the Modal script (`modal_shaped_reward.py`).
- Executes it on an A10G GPU on Modal (~$0.60/hr, run budgeted ~$1).
- Monitors logs, parses the leaderboard, decides based on outcome.
- Iterates automatically: if initial run shows no signal, tries up to 2 reward-shaping variants within budget.
- When a real result lands, updates slide 11 of the Google Slides deck (via Chrome automation) with the new held-out numbers.
- Commits everything. Pushes. Leaves a status note at `/tmp/modal_status.md`.
- **Will not fabricate numbers, push fake results, or spend more than $4 of Modal credit.**

---

## The prompt to paste (copy the fenced block below verbatim)

```
You are resuming a STAT 4830 final project in
/Users/aadithyasrinivasan/Projects/STAT-4830-RL-project. The final
presentation is today at 1 PM. Modal is already set up on this machine
(~/.modal.toml, workspace asrinivasan75, ~$5 free credit available).

# Context you must read before acting
1. Read docs/slides_8_to_12.md -- the current slide narrative.
2. Read report.md section 5 and 6 -- the results we currently claim.
3. Read docs/results/poker_rl_expB_mixed20_20260422/poker_rl_expB_mixed20_20260422/eval_leaderboard.json
   -- the baseline we are trying to beat (20.0% held-out flat across 100 iters).
4. Read scripts/modal_autonomous_prompt.md (this file) for why we are here.

# Your goal
Produce a real held-out-eval curve showing whether a shaped reward
(tool_bonus for real code + penalty for wrapped_action_code fallback)
breaks the 20% flat-line attractor that Exp B diagnosed. Curve becomes
the evidence for a rewritten slide 11.

# The work, in order

## Step 1. Write modal_shaped_reward.py
Create the file at scripts/modal_shaped_reward.py. It must:

- Define a Modal App named "stat4830-shaped-reward".
- Build an image from pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime that
  pip-installs transformers, peft, trl, datasets, accelerate,
  bitsandbytes>=0.43, matplotlib, and uses add_local_dir("..", "/workspace")
  to ship the repo into the container.
- Define a @app.function(gpu="A10G", timeout=3600) that:
    a. cd /workspace
    b. Applies the 3-point shaped-reward patch to src/poker/training.py
       (same patch as notebooks/colab_shaped_reward.ipynb -- copy the
       patches list verbatim). Skip if 'SHAPED REWARD' already present.
    c. Copies
       docs/results/poker_rl_expB_mixed20_20260422/poker_rl_expB_mixed20_20260422/best_by_eval
       to checkpoints/poker_rl_expB_best.
    d. Runs scripts/poker_train.py --phase rl with these args:
       --model ./checkpoints/poker_rl_expB_best
       --seed 20260422 --rl-iterations 30 --rl-batch-size 4
       --rl-lr 5e-6 --rl-sample-temperature 0.2 --rl-top-p 0.9
       --rl-adv-clip 2.0 --ema-gamma 0.9 --max-new-tokens 512
       --rl-output ./checkpoints/poker_rl_shaped
    e. For each checkpoint in ./checkpoints/poker_rl_shaped/iter_{5,10,15,20,25,30},
       loads via src.training.load_model and runs a held-out eval loop:
       20 episodes, seed 20260422, PokerLocalLLMAgent with max_steps=1,
       max_new_tokens=384, temperature=0.1. Record held_out_acc per iter.
    f. Writes experiments/results/modal_shaped_leaderboard.json in the
       container, returns its contents as a Python dict.
- Define @app.local_entrypoint() main() that calls the remote function,
  prints the leaderboard table, writes it to
  experiments/results/modal_shaped_leaderboard.json locally.

## Step 2. Execute
Run `modal run scripts/modal_shaped_reward.py` from the repo root. Let
it stream logs. If it fails during image build or function execution,
fix the error and retry -- up to 3 attempts total. Budget cap: stop if
cumulative Modal spend from this task exceeds $3 (check via
`modal app list` and `modal app logs` -- if uncertain, stop).

## Step 3. Interpret results
Read experiments/results/modal_shaped_leaderboard.json. Compare each
iter's held_out_acc against the baseline 0.20. Categorize outcome:

- STRONG POSITIVE: any iter >= 0.35 AND monotonic-ish climb visible.
- PARTIAL POSITIVE: any iter >= 0.30 OR training log shows
  wrapped_action_code dropping from 3-4/4 toward 1-2/4 even if held-out
  stays flat.
- NULL: all iters in [0.15, 0.25].
- REGRESSION: any iter <= 0.15.

## Step 4. Iteration policy

- On STRONG POSITIVE: stop, go to Step 5.

- On PARTIAL POSITIVE: stop, go to Step 5. The partial signal is itself
  the story: "fixed the tool-use attractor but not the generalization
  gap, here's what the diagnostic counter now looks like."

- On NULL: try ONE variant. Increase the shaping coefficients to
  tool_bonus += 0.6 (not 0.3) and fallback_penalty = 0.4 (not 0.2).
  Rerun Step 2. If still NULL, stop and go to Step 5 with "null result"
  framing.

- On REGRESSION: stop. The shaping is pushing the policy in the wrong
  direction. Go to Step 5 with the honest "reward coefficients too
  aggressive" framing. Do NOT try to "fix" by fabricating or by picking
  a different eval seed.

Do not exceed 2 Modal runs total.

## Step 5. Update slide 11
Use the Chrome browser automation tools (mcp__claude-in-chrome__*) to
open
https://docs.google.com/presentation/d/1E-8aoxvzoAWQLa6jeFPEOd732Sty0eIJ6QbmI-AM3oU/edit?slide=id.g3d85d2019ac_0_23
and update slide 11 with the new held-out curve. If Chrome is not
logged in or the tab is gone, produce a markdown summary for the user
to paste manually (at docs/slide_11_update.md).

Specifically:
- Generate a new figures/poker_rl_shaped_curve.png showing Exp B's flat
  line at 20% vs the shaped-reward curve overlaid.
- Upload the image into slide 11 via clipboard paste
  (osascript -e 'set the clipboard to (read (POSIX file "...") as
  class PNGf)'; then cmd+v in the slide).
- Update the right-side text callout with the new numbers.
- Update the speaker notes to reflect the shaped-reward finding.

## Step 6. Update report.md and self_critique_week15.md
- Report section 5: add the new held-out leaderboard table with the
  outcome classification you assigned.
- Self-critique: add the outcome as a "what we learned by the deadline"
  bullet.
- Do NOT rewrite the reward-hacking narrative -- the Exp B diagnostic
  is independent evidence and stands either way.

## Step 7. Commit and push
Commit message should accurately reflect the outcome. Examples:
- "Shaped reward breaks 20% attractor: iter 20 held-out = X.X%"
- "Partial positive: real_code rate recovers, held-out still flat"
- "Shaped reward null result: same story as Exp B, updated slides"
Push to origin/main.

## Step 8. Leave a status note
Write /tmp/modal_status.md with a short summary:
- outcome classification (strong/partial/null/regression)
- Modal spend
- files changed
- anything requiring user follow-up before the 1 PM presentation

# Hard constraints (do not violate under any circumstances)
1. Do not fabricate held-out numbers. If the eval JSON is missing,
   broken, or suspicious, stop and report. Never invent.
2. Do not update slides, report, or self-critique to claim numbers
   that are not in experiments/results/modal_shaped_leaderboard.json.
3. Do not spend more than $3 of Modal credit cumulatively.
4. Do not make more than 2 Modal training runs.
5. If the upstream src/poker/training.py patch insertion points have
   drifted and patches fail, stop and report. Do not silently disable
   the patch.
6. Do not push commits that break pytest. Run `python -m pytest tests/
   -q --tb=line` before any commit that touches src/.
7. If you cannot complete a step due to missing access (Chrome tab
   closed, Modal credit exhausted, etc.) stop at that step, commit
   what you have, write /tmp/modal_status.md, and exit cleanly.

# Expected time budget
- Write modal_shaped_reward.py: 20 min of your work.
- Modal image build + training + eval: ~45-70 min wall clock.
- Slide + report update: 20 min.
- Total: ~90-110 min from when you start to when status note is written.

Begin. Read the context files first (docs/slides_8_to_12.md, report.md,
docs/results/poker_rl_expB_mixed20_20260422/.../eval_leaderboard.json,
scripts/modal_autonomous_prompt.md). Then write modal_shaped_reward.py.
```

---

## How to use this in the morning

1. Open a fresh Claude Code window in the repo: `claude /Users/aadithyasrinivasan/Projects/STAT-4830-RL-project`
2. Paste the fenced block above. Hit enter.
3. Walk away for 90-110 minutes. Come back to `/tmp/modal_status.md`.

If you want to watch it work, leave the terminal visible. Claude will stream its own commentary as it progresses.

## Before you paste, sanity check

- Your Mac is awake (`caffeinate -dis` in another terminal if sleeping is a risk).
- Chrome is open with the Google Slides deck tab (only needed for Step 5 auto-update; markdown fallback exists).
- `~/.modal.toml` exists and `modal profile current` returns `asrinivasan75`. (It does as of now.)
- You have ~$5 of Modal credit. The run is budgeted to stay under $3.

## If something goes wrong while you sleep

The `/tmp/modal_status.md` note will explain what state things are in. Worst case, the Exp B slides and the reward-hacking narrative are still defensible as-is — the Modal experiment is pure upside.
