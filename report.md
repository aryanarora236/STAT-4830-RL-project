## PROBLEM STATEMENT

We aim to build and optimize a Recursive Language Model (RLM), where a base LLM interacts with an external Python REPL that holds long or complex context. The goal is effective recursive editing: the model must use Python tool calls (searching, transforming, filtering) to retrieve or generate correct answers on tasks where normal long-context prompting fails due to context degradation. This matters because real tasks like log analysis, large-file parsing, and multi-step code tasks require repeated access to external information rather than a single forward pass through the model.

We measure success with four metrics: accuracy (exact match), reward (correctness minus step and token penalties), REPL steps (fewer is better), and runtime. Constraints include a safe execution environment (whitelisted builtins, blocked imports, 5-second timeout) and currently synthetic data only. Risks include LLM-generated code exploiting sandbox gaps, sparse rewards during RL training, and heuristic trajectories being too formulaic for effective imitation learning.

## TECHNICAL APPROACH

### Architecture

The system consists of three components:
1. **Task Generator**: Produces (context, question, answer) tuples across three task families
2. **Agent**: Issues Python code to execute in the REPL, receives stdout/stderr, produces final answer
3. **Training Loop**: Collects trajectories, filters successes, and (in future) performs gradient updates

### Objective Function

We maximize expected reward under policy π:

R = C − λ_s · (T / T_max) − λ_t · N_tokens

Where C ∈ {0,1} is binary correctness, T is the number of REPL steps taken, T_max is the maximum allowed steps, λ_s = 0.05 is the step penalty weight, and λ_t = 0.0001 is the token penalty weight.

### Task Types

| Task | Required Steps | Description |
|------|---------------|-------------|
| Needle-in-haystack | 1 | Find KEY=VALUE in random filler text |
| KV extraction | 2 | Filter structured log by field, extract target field |
| Aggregation | 2 | Find all METRIC_* keys, sum numeric values |

### Agent Implementations

**DeterministicAgent** (Week 4): Single-step regex search. Parses question for key name, generates regex pattern, executes in REPL. Serves as correctness baseline.

**HeuristicMultiStepAgent** (Week 6): Strategy-based agent that classifies the question type and dispatches to the appropriate multi-step strategy. Each strategy generates and executes Python code in 1-2 REPL steps.

**LLMAgent** (Week 8): LLM-based agent using the HuggingFace Inference API (`Qwen/Qwen2.5-Coder-7B-Instruct` by default, configurable). Sends a system prompt instructing the LLM to generate Python code for a sandboxed REPL, along with a context preview and question. Extracts code from markdown blocks, executes via `safe_execute_code`, and uses a multi-step retry loop (up to 5 steps) — if execution fails, the error is sent back to the LLM for correction. Includes exponential backoff for rate limit handling.

### Training Pipeline (Week 7)

1. Trajectory collection: Run agent on batch of tasks, record (context, question, actions, reward)
2. Success filtering: Keep trajectories with reward ≥ threshold for imitation learning
3. Behavior cloning (future): Fine-tune LLM on successful action sequences
4. Reward-weighted updates (future): Weight training examples by reward

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
- 40 trajectories collected, all pass success filter

### Current Limitations
1. LLM accuracy varies by task complexity — multi-step retry helps but is not guaranteed
2. Rate limits on free-tier HuggingFace API constrain throughput
3. Training loop collects trajectories but does not update model parameters
4. Task diversity is limited to 3 synthetic types
5. Strategy selection for heuristic agent is hardcoded via regex, not learned

## SELF-CRITIQUE (Week 8)

### OBSERVE
The pipeline is complete from task generation through reward computation and trajectory collection. All 7 tests pass. However, the core research question — can a model learn to generate effective REPL code? — remains untouched because we have not integrated an LLM.

### ORIENT
**Strengths:** Modular architecture, three working task types, reliable sandbox, training loop ready for data.
**Weaknesses:** No LLM, no actual learning, behind on written deliverables.
**Risk:** LLM integration may surface problems (malformed code, sandbox escapes, latency) not seen with template agents.

### DECIDE
1. Integrate a small LLM over spring break (Qwen-0.5B or API model)
2. Run behavior cloning on heuristic trajectories in Week 9
3. Add more diverse tasks and begin reward-weighted training in Week 10

### ACT
Need GPU access (Colab free tier) or API credits. Need to study rlm-minimal's training code for behavior cloning reference.
