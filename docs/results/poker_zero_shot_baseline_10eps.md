# Poker Zero-Shot Baseline (10 Episodes)

This run is a longer baseline than the 2-episode smoke test.

How this evaluation works:
- Generate a fresh poker scenario `(context, question, heuristic_answer)` each episode.
- `PokerLLMAgent` (zero-shot) receives context and writes Python code for the sandboxed REPL.
- The sandbox executes that code and the agent prints a final action (`fold/check/call/raise`).
- `PokerEvaluationFramework` compares the predicted action to the heuristic label and aggregates:
  - exact match
  - action-type match
  - reward proxy
  - average REPL steps
  - per-action and per-street accuracy

Command used:

`HF_TOKEN=*** python3 -c "from src.poker.agents import PokerLLMAgent, PokerHeuristicAgent; from src.poker.evaluation import PokerEvaluationFramework; from src.poker.tasks import generate_poker_task; llm=PokerLLMAgent(max_steps=2, model_id='Qwen/Qwen2.5-Coder-7B-Instruct', temperature=0.2); heur=PokerHeuristicAgent(); ev=PokerEvaluationFramework(agents=[llm, heur], task_generator=generate_poker_task, num_episodes=10); ev.run_evaluation(); ev.display_results()"`

Output:

```text
==================================================
Agent: PokerLLMAgent
==================================================
Exact match:  2/10 (20.0%)
Type match:   2/10 (20.0%)
Avg reward:   0.200
Avg steps:    2.0

Per-action accuracy:
  call    : 2/4 (50%)
  check   : 0/4 (0%)
  fold    : 0/1 (0%)
  raise   : 0/1 (0%)

Per-street accuracy:
  flop    : 2/3 (67%)
  preflop : 0/5 (0%)
  river   : 0/1 (0%)
  turn    : 0/1 (0%)

==================================================
Agent: PokerHeuristicAgent
==================================================
Exact match:  10/10 (100.0%)
Type match:   10/10 (100.0%)
Avg reward:   1.000
Avg steps:    3.0

Per-action accuracy:
  call    : 4/4 (100%)
  check   : 4/4 (100%)
  fold    : 1/1 (100%)
  raise   : 1/1 (100%)

Per-street accuracy:
  flop    : 3/3 (100%)
  preflop : 5/5 (100%)
  river   : 1/1 (100%)
  turn    : 1/1 (100%)
```
