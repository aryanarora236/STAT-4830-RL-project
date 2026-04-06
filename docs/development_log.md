## Development Log

### Week 4 (Jan 27 – Feb 6)

**Day 1**: Met to understand Week 4 deliverable requirements. After reading the instruction document, we realized the core requirement is having a minimal runnable optimization loop, not necessarily a trained agent. We discussed possible directions and shifted to recursive editing and RLM-style tool use after seeing the professor's guidance. Reviewed the RLM-minimal GitHub repo.

**Day 2**: Spent time understanding the RLM-minimal algorithm. Key takeaway: the model uses a Python REPL as external memory rather than fitting long context into the attention window. Discussed how to reproduce this using a toy needle-in-haystack task. Some early attempts resulted in Python execution errors due to missing sandbox.

**Day 3**: Implemented the `safe_exec` function. First version failed because Python code could still import modules. Added explicit checks to block imports and replaced builtins with a restricted safe dictionary. Confirmed we could evaluate code inside the REPL environment.

**Day 4**: Built the initial needle task generator. Tested with small context sizes (500–2000 characters). Verified needle insertion logic and confirmed regex could reliably extract the value. Wrote first deterministic "tool agent" and ran the first working episode end-to-end.

**Day 5**: Expanded tests to longer contexts (10,000–30,000 characters). Added timing and transcript logging for each REPL call. Built a batch runner to compute accuracy and average runtime. Deterministic agent solved every case.

**Day 6**: Tried integrating a small HuggingFace model but ran into slow setup and model loading issues. Postponed LLM generation to next week. Created a stub function simulating an LLM call.

**Day 7**: Produced final notebook structure with sections for problem setup, mathematical formulation, implementation, validation, and next steps. Reran all cells to confirm nothing breaks. Collected metrics for the report.

### Week 5 (Feb 10 – Feb 13)

Focused on understanding what makes tasks require multi-step reasoning vs. single-step. Read through RLM-minimal's task definitions and identified two additional task families: structured log extraction (filter then extract) and aggregation (findall then compute). Began outlining how a strategy-based agent would select different approaches based on question type.

Started drafting slides outline for the lightning talk but did not complete a full deck.

### Week 6 (Feb 17 – Feb 20)

**Implemented multi-step tasks:**
- `generate_kv_extraction_task`: Creates structured log entries with timestamp, id, level, status fields. Agent must filter by one field (e.g., level=ERROR) and extract another field (e.g., status). Requires 2 REPL steps.
- `generate_multistep_task`: Embeds multiple METRIC_X=<value> pairs in filler text. Agent must find all pairs and sum their numeric values. Requires 2 REPL steps.

**Implemented HeuristicMultiStepAgent:**
- Classifies question type using regex on the question text
- Dispatches to one of three strategies: simple needle, KV filter, aggregation
- Each strategy generates and executes Python code in 1-2 steps
- All tests pass on all three task types

**Key decision**: We chose to build a heuristic agent before LLM integration so we have a "gold standard" trajectory set for future imitation learning.

### Week 7 (Feb 24 – Feb 27)

**Implemented reward function:**
- R = C - λ_s(T/T_max) - λ_t · N_tokens
- Correct + 1 step → reward ≈ 0.995
- Correct + 10 steps → reward ≈ 0.95
- Wrong → reward ≈ -0.005 to -0.05

**Implemented training loop skeleton (TrainingLoop class):**
- `collect_batch()`: Runs agent on N tasks, records trajectories with rewards
- `filter_successes()`: Returns trajectories above reward threshold
- `train_step()`: One iteration of collect + filter + log statistics
- `run()`: Multiple training iterations with progress logging

**Implemented Trajectory class** to store (context, question, answer, actions, reward) per episode.

Updated EvaluationFramework to compute and display reward alongside accuracy and step count.

Revised slides outline and began working on Week 8 deliverables.

### Week 8 (Mar 1 – Mar 7)

Created Week 8 implementation notebook demonstrating the full pipeline: task generation → agent evaluation → reward computation → training loop with trajectory collection.

Updated tests to cover all new functionality (7/7 tests pass: missing needle, multiple needles, long haystack, KV extraction, aggregation, reward computation, training loop).

Prepared lightning talk for Mar 3 class presentation.

**Key open question**: Which LLM to integrate? Options are Qwen-0.5B (local, free, slow) vs. API model (fast, costs money). Planning to decide over spring break.

Integrated LLMAgent using HuggingFace Inference API (Qwen2.5-Coder-7B-Instruct). The agent sends a system prompt + context preview to the LLM, extracts Python code from the response, executes it in the sandbox, and retries on error (up to 5 steps). Also built LocalLLMAgent for local training with LoRA.

Ran the full pipeline on Colab GPU: behavior cloning on heuristic trajectories followed by REINFORCE. Did not save checkpoints or detailed logs — need to rerun with proper tracking.

### Spring Break + Week 9 (Mar 7 – Mar 22)

**Pivoted application domain from synthetic tasks to poker.**

Motivation: The synthetic tasks (needle-in-haystack, KV extraction, aggregation) were too simple to demonstrate the value of long-context reasoning. The heuristic solved everything at 100%, and the "long context" was just filler text — the model didn't need to actually read it. Poker is a better fit because:
- Hand history is genuinely long context that matters for optimal play
- Computing opponent tendencies (VPIP, PFR, aggression) requires parsing previous hands
- The retrieve → compute → decide flow maps naturally to the RLM framework

**Built the poker environment** (`src/poker/environment.py`):
- Card, Deck, HandEvaluator with 5-best-of-7 hand ranking
- Monte Carlo equity estimation (configurable simulations vs. random opponent)
- GameState with full game representation: hole cards, board, stacks, pot, positions, betting history
- Structured HandRecord for previous hands (parseable by code, not just narrative)
- Five opponent archetypes: Rock, TAG, LAG, Fish, Maniac — each with distinct stat profiles (VPIP, PFR, aggression, fold-to-cbet, 3-bet%)

**Built the heuristic bot** (`src/poker/heuristic.py`):
- 3-step retrieve → compute → decide flow:
  1. Parse hand history → compute per-opponent stats (OpponentStats dataclass)
  2. Evaluate hand strength, pot odds, draws
  3. Apply opponent-specific adjustments to base strategy
- Preflop: 5-tier hand ranking, position-aware open/call/fold
- Postflop: hand category (monster/strong/medium/weak/nothing) + draw detection
- Six adjustment types: exploit fish, respect rocks, trap maniacs, bluff high fold-to-cbet, fold to passive bets, call down aggressive players
- Full ReasoningTrace output for trajectory collection

**Built the task generator** (`src/poker/tasks.py`):
- Generates (context, question, answer) tuples with ~3,000 char contexts
- 15-20 hand histories per scenario, generated from consistent opponent profiles
- `generate_poker_task_with_trace()` also returns the full reasoning trace for BC data
- Poker-specific system prompt for the LLM agent

**Testing**: Opponent adjustments fire ~9% of the time across 500 randomly generated scenarios. Confirmed adjustments for all six types (vs. aggressive, passive, tight, loose, high fold-to-cbet, maniac).

### Week 10 (Mar 24 – Mar 27)

Wrote Draft 4 report reflecting the poker pivot. Updated self-critique (OODA format). Discussed project direction with team — agreed on the poker application and the BC → RL training plan.

Upcoming: need to run zero-shot LLM evaluation on poker tasks and begin behavior cloning on Colab.

### Week 11 (Mar 28 – Apr 5)

**Zero-shot LLM baseline established** (done by teammate on Colab):
- Ran Qwen2.5-Coder-7B-Instruct on 25 poker tasks via HuggingFace API
- Result: 8% accuracy (2/25), avg reward 0.092
- Model struggles to parse structured context and generate valid poker actions
- Baseline saved to `docs/results/poker_zero_shot_baseline_25eps.txt`

**Local model agent added** (`PokerLocalLLMAgent`):
- Built by teammate — variant of PokerLLMAgent that uses a locally loaded transformers model
- Supports evaluation and training loops without HuggingFace API dependency
- Added to `src/poker/agents.py`

**Training script added** (`scripts/poker_train.py`):
- Full CLI for BC → RL → Eval pipeline with configurable arguments
- Supports phase selection (bc/rl/full/eval), model selection, hyperparameters
- Built by teammate for Colab training runs

**Heuristic opponent modeling improved** (adjustment rate 9% → 15.4%):
- Added steal attempts vs tight players from late position
- Added wider calling ranges vs fish for cheap calls (tier 4 hands)
- Added calling down vs maniacs wider (tier 3 hands)
- Added TAG-specific adjustment: size up value bets with premiums
- Added thin value bets with weak hands vs loose postflop (threshold lowered to strength ≥ 0.30)
- Added postflop sizing adjustments: larger vs calling stations, smaller vs tight
- Lowered history threshold from 3 to 2 observed hands

**Comprehensive evaluation experiments run** (500 episodes each):
- Heuristic evaluation: 100% accuracy, detailed per-action and per-street breakdown
- Action distribution: fold 31%, call 32%, check 23%, raise 15%
- Postflop hand categories: nothing 36%, weak 29%, medium 15%, strong 15%, monster 5%
- Adjustment types: calling down aggressive (most common at 39), thin value bets (14), fold to passive (7)
- REPL pipeline validation: 100% code execution success across 600 executions (0 errors)
- Trajectory collection: 500 BC trajectories in 0.4s, avg 3,335 chars of code each

**23 poker-specific tests added** (`tests/test_poker.py`):
- Environment: deck, hand evaluator (pair, flush, straight, full house, 5-of-7), game state formatting
- Heuristic: preflop tiers, hand keys, postflop strength, flush/straight draw detection, opponent stats parsing, decision-making
- Tasks: generation, trace generation, history presence, all-streets coverage
- Rewards: action parsing, exact match, type match, wrong action
- Agents: PokerHeuristicAgent 3-step REPL pipeline validation
- Evaluation: framework metrics and confusion matrix
- Total: 34 tests, all passing

**Deliverables**: Draft 5 report (this document), Week 11 self-critique, updated development log, experiment results saved to `experiments/results/week11_results.json`
