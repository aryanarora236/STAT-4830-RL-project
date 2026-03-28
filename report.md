# Report Draft 4 — Week 10 (Mar 27)

## PROBLEM STATEMENT

We build a Recursive Language Model (RLM) system where an LLM interacts with an external Python REPL to make decisions in domains that require long-context reasoning. The core idea: instead of stuffing everything into the attention window, the model writes code to retrieve, compute, and act on information from a large context.

We apply this to **No-Limit Texas Hold'em poker**, where long context matters because optimal play depends on parsing multi-hand histories to identify opponent tendencies. A player who ignores history plays a fixed strategy; a player who reads history can exploit patterns (e.g., an opponent who always folds to continuation bets). This makes poker a natural testbed for whether an RLM can learn to use long context through tool calls.

### Success Metrics
- **Decision accuracy**: Does the model's action match or beat the heuristic baseline?
- **Opponent exploitation**: Does the model adjust its play based on opponent history (not just hand strength)?
- **Reasoning quality**: Does the model follow a retrieve → compute → decide flow?
- **EV improvement**: After RL training, does the model achieve higher expected value than the heuristic?

### Constraints
- Safe execution environment: whitelisted builtins, blocked imports, 5-second timeout per code execution
- Synthetic poker data (procedurally generated hands with consistent opponent profiles)
- Training on small models (Qwen 1.5B with LoRA) due to compute limits

### Risks
- LLM-generated poker analysis code may be incorrect or miss edge cases
- Behavior cloning may teach the model to ignore history if heuristic trajectories are too formulaic
- RL training with sparse rewards (win/lose at showdown) may be unstable
- Opponent profiles in synthetic data may not capture real-world complexity

## TECHNICAL APPROACH

### Architecture

The system has three components:

1. **Poker Environment**: Generates realistic game states with structured hand histories. Each opponent has a consistent archetype (rock, TAG, LAG, fish, maniac) that determines their behavior across hands. The context includes hole cards, board, stacks, pot, betting history, and 15-20 previous hands.

2. **Agent**: Receives the game state as text, writes Python code to analyze it via the REPL, and outputs a decision (fold/check/call/raise). The agent follows a 3-step reasoning flow:
   - **Retrieve**: Parse hand history to compute per-opponent stats (VPIP, PFR, aggression factor, fold-to-cbet)
   - **Compute**: Evaluate hand strength, pot odds, draws, and equity
   - **Decide**: Combine hand analysis with opponent profile to choose an action

3. **Training Pipeline**: Behavior cloning on heuristic trajectories (Phase 1), then REINFORCE with EV-based rewards (Phase 2).

### Poker Environment

The environment generates (context, question, answer) tuples:

- **Context**: ~3,000 characters of structured text including the current hand state and 15-20 previous hand records. Opponent actions in history are generated probabilistically from their archetype profiles, ensuring consistency across hands.
- **Question**: "What should you do?" with instructions to parse history before deciding.
- **Answer**: The heuristic bot's decision (used as ground truth for behavior cloning).

Five opponent archetypes with distinct statistical profiles:

| Archetype | VPIP | PFR | Aggression | Fold-to-CBet |
|-----------|------|-----|------------|-------------|
| Rock | 14% | 11% | 1.2 | 70% |
| TAG | 22% | 18% | 2.0 | 55% |
| LAG | 35% | 28% | 2.8 | 40% |
| Fish | 52% | 10% | 0.6 | 35% |
| Maniac | 60% | 42% | 3.5 | 25% |

### Heuristic Baseline (TAG Bot with Opponent Modeling)

The heuristic serves as both the baseline to beat and the teacher for behavior cloning. It follows the same 3-step flow the LLM must learn:

**Step 1 — Retrieve**: Parses all previous hand records and computes per-opponent stats:
- VPIP (voluntarily put money in pot)
- PFR (preflop raise frequency)
- Postflop aggression factor (bets+raises / calls)
- Fold-to-continuation-bet percentage

**Step 2 — Compute**:
- Preflop: Categorizes hand into 5 tiers (AA/KK/QQ/AKs = Tier 1 → trash = Tier 5)
- Postflop: Evaluates made hand (monster/strong/medium/weak/nothing), detects flush and straight draws, computes pot odds

**Step 3 — Decide**: Applies opponent-specific adjustments to the base strategy:
- vs. Fish (loose-passive): value bet wider, raise for value with Tier 2+ hands
- vs. Rock (tight): fold marginal hands to their raises
- vs. Maniac (loose-aggressive): trap with premiums, call down wider
- vs. High fold-to-cbet: bluff on flops with weak hands
- vs. Passive player who bets: fold medium hands (they usually have it)
- vs. Aggressive player: call down with medium/weak hands (they may be bluffing)

### Objective Function

**Phase 1 (Behavior Cloning)**: Supervised cross-entropy loss on heuristic trajectories. The model learns to replicate the retrieve → compute → decide flow.

**Phase 2 (Reinforcement Learning)**: The model plays hands against the heuristic bot. Reward is based on decision quality:

$$R = \text{EV}(\text{action}) - \lambda_s \frac{T}{T_{\max}} - \lambda_t \cdot N_{\text{tokens}}$$

Where EV is estimated by simulating the hand outcome, λ_s = 0.05 penalizes excess REPL steps, and λ_t = 0.0001 penalizes token usage.

### Training Pipeline

Built on the existing infrastructure from Weeks 4-8 (trajectory collection, safe sandbox, reward computation), adapted for poker:

1. **Trajectory collection**: Run heuristic bot on generated poker scenarios, record full reasoning traces (retrieve step + compute step + decision)
2. **Behavior cloning**: Fine-tune Qwen 1.5B with LoRA on successful trajectories using SFT
3. **REINFORCE**: Policy gradient training where the model plays against the heuristic, reward = chips won/lost

### Hand Evaluation

The hand evaluator supports standard poker hand rankings (high card through straight flush) with best-5-of-7 selection. Monte Carlo equity estimation runs configurable simulations against random opponent hands to estimate win probability.

## RESULTS

### Poker Environment Validation

The poker environment generates realistic scenarios across all four streets:

| Street | Avg Context Length | Avg History Hands | Components |
|--------|-------------------|-------------------|------------|
| Preflop | ~2,800 chars | 15 | Hole cards, stacks, positions, history |
| Flop | ~3,100 chars | 15 | + 3 community cards, flop betting |
| Turn | ~3,200 chars | 15 | + 4 community cards, turn betting |
| River | ~3,300 chars | 15 | + 5 community cards, river betting |

### Heuristic Bot Performance

Tested across 500 randomly generated scenarios:

- Opponent adjustment rate: ~9% of decisions are modified by opponent history
- Adjustment types observed: exploit aggressive players (call down wider), bluff high fold-to-cbet opponents, fold to passive player bets, value bet wider vs. fish, respect tight player raises, trap maniacs

Example adjustments from testing:

| Scenario | Base Decision | Adjusted Decision | Reason |
|----------|--------------|-------------------|--------|
| Medium pair vs. passive bettor | Call | Fold | Passive player bets = likely has it (Agg=0.5) |
| Weak hand vs. aggressive raiser | Fold | Call | Aggressive opponent may be bluffing (Agg=13.0) |
| Nothing on flop vs. high fold-to-cbet | Check | Bet 50% pot | Opponent folds to c-bets 67% of the time |
| KQs vs. maniac preflop | Raise | Call | Trap the maniac, let them keep betting (Agg=34.0) |

### Previous Results (Weeks 4-8, Synthetic Tasks)

These results from the original synthetic task domain validated the RLM framework before the poker pivot:

| Agent | Task Type | Accuracy | Avg Steps | Avg Reward |
|-------|-----------|----------|-----------|------------|
| DeterministicAgent | Needle | 100% | 1.0 | 0.995 |
| HeuristicMultiStepAgent | Needle | 100% | 1.0 | 0.990 |
| HeuristicMultiStepAgent | KV extraction | 100% | 2.0 | 0.980 |
| HeuristicMultiStepAgent | Aggregation | 100% | 2.0 | 0.980 |

### What We Have vs. What's Planned

| Component | Status |
|-----------|--------|
| Poker environment + task generator | Done |
| Hand evaluation (5-of-7, equity) | Done |
| Heuristic bot with opponent modeling | Done |
| Structured hand history generation | Done |
| 3-step reasoning traces | Done |
| LLM agent (zero-shot evaluation) | In progress |
| Behavior cloning on poker trajectories | Planned (Week 11) |
| REINFORCE with EV reward | Planned (Weeks 12-13) |
| Final evaluation vs. heuristic | Planned (Week 14) |

## CURRENT LIMITATIONS

1. **Heuristic adjustments fire ~9% of the time** — most decisions are made on hand strength alone. The model may learn to ignore history if BC data doesn't contain enough adjustment examples.
2. **No LLM evaluation yet on poker tasks** — we need zero-shot baseline numbers before training.
3. **Opponent profiles are synthetic** — real opponents have more nuanced and inconsistent behavior.
4. **Hand evaluation is slow for equity** — Monte Carlo with 1000 simulations per hand. May need to reduce for training throughput.
5. **EV-based reward for RL is not yet implemented** — currently using correctness-based reward from the original framework.

## NEXT STEPS

### Week 11 (Apr 3)
- Run zero-shot LLM evaluation on poker tasks (accuracy, reasoning quality)
- Collect 500+ heuristic trajectories with full reasoning traces
- Begin behavior cloning training on Colab GPU

### Week 12 (Apr 10)
- Evaluate BC model against heuristic baseline
- Implement EV-based reward function (simulate hands, measure profit)
- Begin REINFORCE training

### Weeks 13-14 (Apr 17-23)
- Full RL training run
- Final evaluation: heuristic vs. zero-shot LLM vs. BC model vs. RL model
- Prepare final presentation showing the improvement arc

### Week 15 (Apr 28)
- Final report with complete results
- Google Colab demo notebook
