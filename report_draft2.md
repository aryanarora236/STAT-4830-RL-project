# Report Draft 2 — Week 6 (Feb 20)

## PROBLEM STATEMENT

We aim to build and optimize a Recursive Language Model (RLM), where a base LLM interacts with an external Python REPL that holds long or complex context. The goal is effective recursive editing: the model must use Python tool calls (searching, transforming, filtering) to retrieve or generate correct answers on tasks where normal long-context prompting fails due to context degradation. This matters because real tasks like log analysis, large-file parsing, and multi-step code tasks require repeated access to external information rather than a single forward pass through the model.

We measure success in two phases: (1) reproducing the RLM-minimal long-context retrieval task to verify the REPL loop works end-to-end, and (2) improving accuracy and efficiency through training (fewer REPL steps, fewer tokens, higher retrieval success). Metrics include accuracy, REPL turns, reward score, and token usage. Constraints include a safe execution environment that prevents imports, file access, and long runtime, and we currently operate on small synthetic datasets.

**Updates since Draft 1:**
- Extended beyond single needle-in-haystack to multi-step tasks (KV extraction, aggregation)
- Defined formal reward function: R = C - λ_s(T/T_max) - λ_t · N_tokens
- Began designing strategy-based agent architecture

## TECHNICAL APPROACH

We implement an RLM as a wrapper around a base LLM with a Python REPL holding the external context. The model issues REPL actions (a1…aT) and a final answer y; reward is correctness minus small penalties for excessive steps or tokens.

### Objective Function

$$R = C - \lambda_s \frac{T}{T_{\max}} - \lambda_t \cdot N_{\text{tokens}}$$

Where C is binary correctness, T is REPL steps, and the λ terms penalise inefficiency.

### Task Types

We now support three task families:
1. **Needle-in-haystack** (1 step): Find KEY=VALUE in random text
2. **KV extraction** (2 steps): Filter structured log lines by one field, extract another field
3. **Aggregation** (2 steps): Find all METRIC_* keys, sum their numeric values

### Agent Architecture

- **DeterministicAgent**: Baseline using regex (Week 4, 100% accuracy on needle tasks)
- **HeuristicMultiStepAgent**: Strategy-based agent that selects approach based on question type, supports multi-step REPL interaction

Training will follow two stages: behavior cloning on successful rollouts, then RL-style optimization (reward-weighted updates) to encourage efficient tool use.

## RESULTS

### Deterministic Baseline (Week 4)
- Accuracy: 100% on needle-in-haystack tasks (10/10 episodes)
- Average runtime: 0.000122 seconds per episode
- Average REPL steps: 1.00

### HeuristicMultiStepAgent (Week 6)
- Successfully handles KV extraction tasks (2 REPL steps)
- Successfully handles aggregation tasks (2 REPL steps)
- All edge cases pass (missing needles, multiple needles, long haystacks)

### Current Limitations
- No LLM-generated tool actions yet (agents use templates)
- Strategy selection is rule-based (regex on question text)
- Training loop not yet implemented
- Limited to 3 synthetic task types

## NEXT STEPS

**Immediate (Weeks 7-8):**
1. Implement reward computation and trajectory collection
2. Build training loop skeleton
3. Prepare lightning talk slides

**Short-term (Weeks 9-10):**
1. Integrate a small local or API LLM to generate Python actions
2. Implement behavior cloning using successful trajectories
3. Add more complex tasks

**Long-term (Weeks 11-15):**
1. Implement RL-style reward-weighted policy gradient
2. Evaluate on real-world tasks
3. Compare with baseline long-context prompting
