# Report Draft 5 — Week 12 (Apr 10)

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

- **Context**: ~4,300 characters of structured text (range 2,800–5,600) including the current hand state and 15 previous hand records. Opponent actions in history are generated probabilistically from their archetype profiles, ensuring consistency across hands.
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
- vs. Fish (loose-passive): value bet wider, widen calling range for cheap calls
- vs. Rock (tight): fold marginal hands to their raises, steal from late position
- vs. Maniac (loose-aggressive): trap with premiums, call down wider
- vs. High fold-to-cbet: bluff on flops with weak hands
- vs. Passive player who bets: fold medium/weak hands (they usually have it)
- vs. Aggressive player: call down with medium/weak hands (they may be bluffing)
- vs. TAG: size up value bets with premium hands
- vs. Loose players postflop: thin value bet with medium+ hands
- Sizing adjustments: larger vs calling stations, smaller vs tight players

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

### Zero-Shot LLM Baseline (Qwen-7B)

Before any training, we evaluated Qwen2.5-Coder-7B-Instruct on 25 poker tasks via the HuggingFace API:

| Metric | Value |
|--------|-------|
| Exact match accuracy | 8.0% (2/25) |
| Type match accuracy | 8.0% (2/25) |
| Average reward | 0.092 |
| Average steps | 1.9 |

Per-action breakdown:
| Action | Accuracy |
|--------|----------|
| Call | 1/15 (7%) |
| Check | 0/4 (0%) |
| Fold | 1/1 (100%) |
| Raise | 0/5 (0%) |

Per-street breakdown:
| Street | Accuracy |
|--------|----------|
| Preflop | 1/3 (33%) |
| Flop | 0/9 (0%) |
| Turn | 1/7 (14%) |
| River | 0/6 (0%) |

The zero-shot model struggles badly — 8% accuracy vs the heuristic's 100%. This establishes a clear baseline that BC training needs to improve on.

### Heuristic Bot Performance (500 episodes)

Tested across 500 randomly generated scenarios:

**Action distribution:**
| Action | Count | Percentage |
|--------|-------|-----------|
| Call | 158 | 31.6% |
| Fold | 155 | 31.0% |
| Check | 114 | 22.8% |
| Raise | 73 | 14.6% |

**Per-street action breakdown:**
| Street | Most common | 2nd | 3rd | 4th |
|--------|-------------|-----|-----|-----|
| Preflop | Check 52% | Fold 38% | Raise 8% | Call 2% |
| Flop | Fold 39% | Call 31% | Check 22% | Raise 9% |
| Turn | Call 45% | Fold 25% | Raise 17% | Check 13% |
| River | Call 48% | Raise 24% | Fold 23% | Check 5% |

**Hand category distribution (postflop):**
| Category | Count | Percentage |
|----------|-------|-----------|
| Nothing | 135 | 36.3% |
| Weak | 109 | 29.3% |
| Medium | 54 | 14.5% |
| Strong | 55 | 14.8% |
| Monster | 19 | 5.1% |

### Opponent Adjustment Rate

**Improved from 9% → 15.4%** through Week 11 heuristic enhancements:

- Added steal attempts vs tight players from late position
- Added wider calling ranges vs fish for cheap calls
- Added calling down vs maniacs wider (tier 3 hands)
- Added thin value bets with medium/weak hands vs loose postflop
- Added sizing adjustments (larger vs calling stations, smaller vs tight)
- Lowered history threshold from 3 to 2 hands

Most common adjustment types (from 500 episodes):
| Adjustment | Count |
|-----------|-------|
| Calling down vs aggressive player | 39 |
| Thin value bet vs loose player | 14 |
| Folding to passive player bet | 7 |
| Stealing vs tight player | 3 |
| Folding to tight player raise | 4 |
| Trapping maniac | 1 |

### REPL Pipeline Validation (200 episodes)

| Metric | Value |
|--------|-------|
| Action type accuracy | 100% (200/200) |
| Code execution success | 100% (600/600) |
| Errors | 0 |
| Stats parsed in output | 100% (200/200) |
| Steps per episode | 3.0 |

### BC Trajectory Collection (500 episodes)

| Metric | Value |
|--------|-------|
| Trajectories collected | 500 |
| Correct | 500/500 (100%) |
| Has code | 500/500 |
| Parsed stats | 500/500 (100%) |
| Avg code length | 3,335 chars |
| Context length | avg 4,288 (range 2,812–5,602) |
| Collection time | 0.4s |

### Previous Results (Weeks 4-8, Synthetic Tasks)

These results from the original synthetic task domain validated the RLM framework before the poker pivot:

| Agent | Task Type | Accuracy | Avg Steps | Avg Reward |
|-------|-----------|----------|-----------|------------|
| DeterministicAgent | Needle | 100% | 1.0 | 0.995 |
| HeuristicMultiStepAgent | Needle | 100% | 1.0 | 0.990 |
| HeuristicMultiStepAgent | KV extraction | 100% | 2.0 | 0.980 |
| HeuristicMultiStepAgent | Aggregation | 100% | 2.0 | 0.980 |

### Test Coverage

| Test Suite | Tests | Status |
|-----------|-------|--------|
| Basic (Weeks 4-8) | 11 | All passing |
| Poker (Week 11) | 23 | All passing |
| **Total** | **34** | **All passing** |

### What We Have vs. What's Planned

| Component | Status |
|-----------|--------|
| Poker environment + task generator | Done |
| Hand evaluation (5-of-7, equity) | Done |
| Heuristic bot with opponent modeling | Done (improved Week 11) |
| Structured hand history generation | Done |
| 3-step reasoning traces | Done |
| Zero-shot LLM evaluation | Done (8% baseline) |
| Local model agent (PokerLocalLLMAgent) | Done |
| Training script (BC + RL + eval) | Done |
| 23 poker-specific tests | Done |
| Behavior cloning on poker trajectories | In progress (Week 12) |
| REINFORCE with EV reward | Planned (Week 13) |
| Final evaluation vs. heuristic | Planned (Week 14) |

## CURRENT LIMITATIONS

1. **Zero-shot LLM accuracy is 8%** — the model cannot play poker without training. Most errors come from failing to parse the context or outputting malformed actions.
2. **Preflop hands are 84% Tier 5** — the model may learn to default to fold/check. We should verify BC doesn't collapse to this.
3. **Opponent adjustments fire 15.4% of the time** — improved from 9%, but the model may still learn to ignore history if adjustment examples are rare in BC data.
4. **Opponent profiles are synthetic** — real opponents have more nuanced and inconsistent behavior.
5. **Hand evaluation is slow for equity** — Monte Carlo with 1000 simulations per hand. May need to reduce for training throughput.

## NEXT STEPS

### Week 12 (Apr 10)
- Run BC training on Colab GPU (500 trajectories, Qwen 1.5B + LoRA, 3 epochs)
- Evaluate BC model vs zero-shot (8%) and heuristic (100%)
- Key question: does BC model learn to parse opponent stats?

### Week 13 (Apr 17)
- Implement EV-based reward for RL (simulate hands, measure profit)
- Begin REINFORCE training from BC checkpoint
- Final evaluation: zero-shot vs BC vs RL vs heuristic

### Week 14 (Apr 21-23)
- Final presentations
- Complete evaluation across all agent types and all streets
- Prepare demo notebook showing the full improvement arc

### Week 15 (Apr 28)
- Final report with complete results
- Google Colab demo notebook
