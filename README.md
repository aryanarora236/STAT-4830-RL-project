# STAT 4830 Final Project — Recursive Language Models for Poker

Fine-tune a small open-weight LLM to play No-Limit Texas Hold'em by writing Python code that parses multi-hand histories, computes opponent statistics, and selects an action. The project applies the Recursive Language Model (RLM) pattern — language model + sandboxed Python REPL — to a domain where long structured context actually changes the decision.

**Course:** STAT 4830 (Spring 2026), Prof. Damek Davis
**Team:** Aadithya Srinivasan, Aryan Arora, Aarav M.

## Headline result

Training-time rollout accuracy during REINFORCE fine-tuning from the BC checkpoint, on Qwen2.5-Coder-1.5B with LoRA (batch 4, 130 iterations total):

| Milestone | EMA accuracy | Single-batch peak |
|---|---|---|
| BC initialization (first RL iter) | 25.0% | — |
| Long RL run peak (iter 105) | **42.3%** | **75.0%** |
| Long RL run final (iter 120) | 28.7% (drift) | — |
| Heuristic (ground truth) | 100% | 100% |
| Zero-shot Qwen-7B (25-ep eval, HF Inference API) | 8.0% | — |

Held-out eval on the iter_105 checkpoint produces `experiments/results/final_eval_iter105.json`:

| Agent | All streets | Preflop | Postflop | Avg reward |
|---|---|---|---|---|
| Zero-shot Qwen-1.5B | [ZS_ALL]% | [ZS_PRE]% | [ZS_POST]% | [ZS_R] |
| BC Qwen-1.5B | [BC_ALL]% | [BC_PRE]% | [BC_POST]% | [BC_R] |
| RL Qwen-1.5B (iter 105) | [RL_ALL]% | [RL_PRE]% | [RL_POST]% | [RL_R] |
| Heuristic | 100% | 100% | 100% | 1.000 |

Full methodology, hyperparameter tables, training curves, confusion matrices, and a detailed discussion of the reward-hacking failure mode observed over the last 15 iterations are in [`report.md`](report.md). Final self-critique in [`self_critique_week15.md`](self_critique_week15.md).

## 30-second summary

- **Task:** given a poker game state + 15 prior hand records (~4k characters), output one of `fold / check / call $X / raise $X`.
- **Agent:** emits Python code that runs in a sandboxed REPL with `CONTEXT` as a string global. The last printed line is parsed as the action.
- **Training:** (1) behavior cloning on 500 heuristic-generated `(prompt, code)` pairs; (2) REINFORCE with batch-normalized clipped advantages on the BC warm-start.
- **Compute:** ~90 minutes end-to-end on a single H100 80GB.

## Reproducing the results

### Option A — Colab demo (fastest; no training)

Open [`notebooks/final_demo.ipynb`](notebooks/final_demo.ipynb) in Google Colab, switch the runtime to T4 GPU, and `Run All`. This clones the repo, downloads the published RL checkpoint, runs 3 demo scenarios with full reasoning trace, and prints the final comparison table. ~5 minutes end-to-end.

### Option B — Full training on PrimeIntellect (~90 min, ~$3–5)

1. Install CLI: `curl -fsSL https://get.primeintellect.ai | sh && prime login`
2. Pick an H100 80GB: `prime availability list --gpu-type H100_80GB`
3. Launch: `prime pods create --id <availability-id> --env HF_TOKEN=$HF_TOKEN --name stat4830`
4. SSH: `prime pods ssh <pod-id>`
5. Bootstrap + run on the pod:
   ```bash
   INSTALL_UNSLOTH=1 bash <(curl -sSL https://raw.githubusercontent.com/aryanarora236/STAT-4830-RL-project/main/scripts/primeintellect/bootstrap.sh)
   cd /root/STAT-4830-RL-project && source .venv/bin/activate
   bash scripts/primeintellect/run_full.sh
   ```
6. `scp` results back. Full playbook with troubleshooting: [`scripts/primeintellect/README.md`](scripts/primeintellect/README.md).

### Option C — Local (eval + small-scale BC only; RL needs a real GPU)

```bash
git clone https://github.com/aryanarora236/STAT-4830-RL-project.git
cd STAT-4830-RL-project
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/ -q                        # 36 tests, ~2 seconds
python scripts/poker_train.py --phase eval \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --eval-episodes 10 --eval-by-street              # zero-shot eval
```

For BC and RL you need an Ampere+ GPU with ≥24 GB VRAM (A10G, A100, L40, H100).

## Repository layout

```
├── report.md                          # Final conference-style report
├── self_critique_week15.md            # Final self-critique
├── README.md                          # This file
├── requirements.txt                   # Python deps
├── src/
│   ├── models.py                      # Agent base class, LLMAgent, sandbox helpers
│   ├── training.py                    # load_model, BC trainer, REINFORCE trainer, set_training_seed
│   ├── utils.py                       # safe_execute_code, original synthetic task generators
│   └── poker/
│       ├── environment.py             # GameState, HandRecord, 5-of-7 evaluator, opponent archetypes
│       ├── heuristic.py               # HeuristicPokerBot (TAG w/ opponent modeling)
│       ├── tasks.py                   # Task generators (all/preflop/postflop) + system prompt
│       ├── agents.py                  # PokerHeuristicAgent, PokerLLMAgent, PokerLocalLLMAgent
│       ├── rewards.py                 # parse_action, compute_poker_reward_simple
│       ├── training.py                # PokerBCTrainer, PokerReinforceTrainer
│       └── evaluation.py              # PokerEvaluationFramework + JSON export
├── scripts/
│   ├── poker_train.py                 # CLI: --phase {bc,rl,full,eval}
│   ├── train.py                       # Original synthetic-task pipeline (Week 4-8)
│   ├── fill_report_from_eval.py       # Fills [BC_ALL], [RL_PRE], ... from final_eval.json
│   └── primeintellect/
│       ├── README.md                  # Pod launch + SSH + bootstrap + retrieve playbook
│       ├── bootstrap.sh               # Idempotent pod env setup
│       └── run_full.sh                # Single-command pipeline runner
├── tests/
│   ├── test_basic.py                  # 11 tests: synthetic tasks + agent + training
│   └── test_poker.py                  # 25 tests: poker env, heuristic, tasks, eval, BC mix, JSON roundtrip
├── notebooks/
│   ├── final_demo.ipynb               # Colab demo (load RL checkpoint + eval)
│   ├── week12_bc_training.ipynb       # BC training smoke test
│   └── week{4,8,10}_*.ipynb           # Earlier deliverables
├── experiments/
│   ├── results/                       # final_eval_*.json, week11_results.json, week12_analysis.json
│   ├── week11_evaluation.py           # Reproducible 500-episode eval
│   └── week12_analysis.py             # BC dataset + ablation analysis
├── figures/                           # Plots (reward flowchart, training curves, Week 12 analysis)
└── docs/
    ├── development_log.md             # Weekly progress log
    ├── results/                       # Run logs + per-run manifests
    ├── final_slides_outline.md        # Final presentation slide content
    └── assignments/                   # Week 4 deliverable instructions (from project base)
```

## Key files for graders

| Looking for… | Read |
|---|---|
| Problem statement + methodology + results | [`report.md`](report.md) |
| Final self-critique (Week 15) | [`self_critique_week15.md`](self_critique_week15.md) |
| Prior self-critiques | `self_critique_week{6,8,10,11}.md` |
| Development log (15 weeks) | [`docs/development_log.md`](docs/development_log.md) |
| Executable demo | [`notebooks/final_demo.ipynb`](notebooks/final_demo.ipynb) |
| Final presentation | [`docs/final_slides_outline.md`](docs/final_slides_outline.md) + `Final Presentation/*.pptx` |
| Training pipeline entry point | [`scripts/poker_train.py`](scripts/poker_train.py) |
| REINFORCE implementation | [`src/poker/training.py`](src/poker/training.py) |
| Heuristic baseline | [`src/poker/heuristic.py`](src/poker/heuristic.py) |
| Structured eval output | `experiments/results/final_eval_*.json` |

## Testing

```bash
python -m pytest tests/ -v
# 36 passed in ~2s
```

Tests cover: card dealing, hand evaluation (pair/flush/straight/full house/5-of-7), game state formatting, preflop tiers, postflop strength, flush/straight draw detection, opponent stats parsing, heuristic decision making, task generation, reasoning traces, action parsing, reward computation, agent REPL pipelines, evaluation metrics, confusion matrices, BC task-mix regression, and eval JSON export roundtrip.

## License

Course project, no license claimed. Built on top of `Qwen2.5-Coder-1.5B-Instruct` (Apache 2.0) and `trl` / `peft` / `transformers` (Apache 2.0).
