# Poker Zero-Shot Baseline (2 Episodes)

Command run:

`HF_TOKEN=*** python3 -c "from src.poker.agents import PokerLLMAgent, PokerHeuristicAgent; from src.poker.evaluation import PokerEvaluationFramework; from src.poker.tasks import generate_poker_task; llm=PokerLLMAgent(max_steps=2, model_id='Qwen/Qwen2.5-Coder-7B-Instruct', temperature=0.2); heur=PokerHeuristicAgent(); ev=PokerEvaluationFramework(agents=[llm, heur], task_generator=generate_poker_task, num_episodes=2); ev.run_evaluation(); ev.display_results()"`

Output:

```text
==================================================
Agent: PokerLLMAgent
==================================================
Exact match:  0/2 (0.0%)
Type match:   0/2 (0.0%)
Avg reward:   0.300
Avg steps:    2.0

Per-action accuracy:
  check   : 0/2 (0%)

Per-street accuracy:
  preflop : 0/2 (0%)

==================================================
Agent: PokerHeuristicAgent
==================================================
Exact match:  2/2 (100.0%)
Type match:   2/2 (100.0%)
Avg reward:   1.000
Avg steps:    3.0

Per-action accuracy:
  check   : 2/2 (100%)

Per-street accuracy:
  preflop : 2/2 (100%)
```

Notes:
- This is a quick smoke-test baseline with only 2 episodes.
- Use a larger run (e.g., 50-100 episodes) for report-grade results.
