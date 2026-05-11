# RL Poker Experiment Analysis (All Runs)

## Included runs
- long_simple_120
- medium_simple_10
- smoke_graph_3
- expA_partial_50 (held-out eval)
- expB_mixed20_100 (held-out eval)
- fast_mixed20_20 (held-out eval)

## Key findings
1. Held-out eval identifies true quality; train metrics alone are noisy.
2. Exp B is flat on held-out eval (0.20/0.20 throughout).
3. Exp A shows modest held-out gains (best 0.25 acc / 0.285 reward).
4. Fast run shows strongest peak (0.40 acc / 0.46 reward at iter 10) but regresses by iter 20 (0.25 / 0.325).
5. Conclusion: RL can improve policy at selected checkpoints, but stability remains the main challenge.

## Files generated
- experiment_summary.csv
- train_reward_over_time.png
- train_accuracy_over_time.png
- heldout_eval_accuracy_over_time.png
- heldout_eval_reward_over_time.png
- best_heldout_comparison.png
