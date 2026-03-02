# Report Draft 3 — Week 8 (Mar 7)

## PROBLEM STATEMENT

We aim to build and optimize a Recursive Language Model (RLM), where a base LLM interacts with an external Python REPL that holds long or complex context. The goal is effective recursive editing: the model must use Python tool calls (searching, transforming, filtering) to retrieve or generate correct answers on tasks where normal long-context prompting fails due to context degradation. This matters because real tasks like log analysis, large-file parsing, and multi-step code tasks require repeated access to external information rather than a single forward pass through the model.

### Success Metrics
- **Accuracy**: Percentage of correct final answers (exact match)
- **Reward**: R = C - λ_s(T/T_max) - λ_t · N_tokens, balancing correctness vs. efficiency
- **REPL steps**: Number of tool interactions (fewer is better)
- **Runtime**: Wall-clock time per episode

### Constraints
- Safe execution environment: whitelisted builtins, blocked dangerous imports, 5-second timeout
- Synthetic data for development; real-world tasks planned for later phases
- Local compute for now (small models, CPU-only); GPU via Colab for future LLM integration

### Risks
- LLM-generated code may be malformed or exploit sandbox gaps
- Sparse rewards during RL training may cause instability
- Heuristic trajectories may be too formulaic for effective imitation learning

## TECHNICAL APPROACH

### Architecture

The system consists of three components:
1. **Task Generator**: Produces (context, question, answer) tuples across three task families
2. **Agent**: Issues Python code to execute in the REPL, receives stdout/stderr, produces final answer
3. **Training Loop**: Collects trajectories, filters successes, and (in future) performs gradient updates

### Objective Function

$$\max_\pi \; E\left[ R \mid \pi \right] \quad \text{where} \quad R = C - \lambda_s \frac{T}{T_{\max}} - \lambda_t \cdot N_{\text{tokens}}$$

- $C \in \{0, 1\}$: binary correctness
- $T$: REPL steps taken, $T_{\max}$: maximum allowed steps
- $\lambda_s = 0.05$: step penalty weight
- $\lambda_t = 0.0001$: token penalty weight

### Task Types

| Task | Required Steps | Description |
|------|---------------|-------------|
| Needle-in-haystack | 1 | Find KEY=VALUE in random filler text |
| KV extraction | 2 | Filter structured log by field, extract target field |
| Aggregation | 2 | Find all METRIC_* keys, sum numeric values |

### Agent Implementations

**DeterministicAgent** (Week 4): Single-step regex search. Parses question for key name, generates regex pattern, executes in REPL. Serves as correctness baseline.

**HeuristicMultiStepAgent** (Week 6): Strategy-based agent that classifies the question type (needle / KV filter / aggregation) and dispatches to the appropriate multi-step strategy. Each strategy generates and executes Python code in 1-2 REPL steps.

### Training Pipeline (Week 7)

1. **Trajectory collection**: Run agent on batch of tasks, record (context, question, actions, reward)
2. **Success filtering**: Keep trajectories with reward ≥ threshold for imitation learning
3. **Behavior cloning** (future): Fine-tune LLM on successful action sequences
4. **Reward-weighted updates** (future): Weight training examples by reward to encourage efficient strategies

## RESULTS

### Evaluation Summary

| Agent | Task Type | Accuracy | Avg Steps | Avg Reward |
|-------|-----------|----------|-----------|------------|
| DeterministicAgent | Needle | 100% | 1.0 | 0.995 |
| HeuristicMultiStepAgent | Needle | 100% | 1.0 | 0.990 |
| HeuristicMultiStepAgent | KV extraction | 100% | 2.0 | 0.980 |
| HeuristicMultiStepAgent | Aggregation | 100% | 2.0 | 0.980 |

### Edge Cases (7/7 pass)
- Missing needle: correctly returns "Needle not found"
- Multiple needles: retrieves correct value
- Long haystack (50 sentences): sub-millisecond retrieval
- Large log (100 entries): 2-step extraction works
- Many metrics (5 keys): correct summation

### Training Loop Statistics (5 iterations, batch=8)
- Accuracy: 100% across all iterations (deterministic agent)
- Average reward: 0.95
- Trajectory buffer: 40 episodes collected
- All trajectories pass success filter (reward ≥ 0.5)

### Current Limitations
1. **No LLM integration** — all agents use templated code, not generated code
2. **Training loop is a skeleton** — collects trajectories but does not update model parameters
3. **Task diversity is limited** — 3 synthetic types, no real-world data
4. **Strategy selection is hardcoded** — regex matching on question text, not learned

## NEXT STEPS

### Spring Break (Mar 7-15)
- Select and integrate a small LLM (Qwen-0.5B locally or API model)
- Test LLM code generation on needle tasks within the safe sandbox
- Set up Google Colab notebook for GPU-accelerated experiments

### Weeks 9-10
- Implement behavior cloning: fine-tune LLM on heuristic agent trajectories
- Add retry/error-recovery logic for malformed LLM-generated code
- Expand task types (multi-condition filtering, transform-then-answer)

### Weeks 11-13
- Move from behavior cloning to reward-weighted policy gradient
- Add token counting and optimise for efficiency
- Evaluate on semi-realistic tasks (log parsing, code search)

### Weeks 14-15
- Final evaluation and comparison with long-context baseline
- Polish report and prepare final presentation
