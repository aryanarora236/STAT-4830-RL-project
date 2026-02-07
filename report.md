PROBLEM STATEMENT

We aim to build and optimize a Recursive Language Model (RLM), where a base LLM interacts with an external Python REPL that holds long or complex context. The goal is effective recursive editing: the model must use Python tool calls (searching, transforming, filtering) to retrieve or generate correct answers on tasks where normal long-context prompting fails due to context degradation. This matters because real tasks like log analysis, large-file parsing, and multi-step code tasks require repeated access to external information rather than a single forward pass through the model.

We measure success in two phases: (1) reproducing the RLM-minimal long-context retrieval task to verify the REPL loop works end-to-end, and (2) improving accuracy and efficiency through training (fewer REPL steps, fewer tokens, higher retrieval success). Metrics that we measure include accuracy, REPL turns, and token usage. Constraints include having a working minimal implementation today, small synthetic datasets, and a safe execution environment that prevents imports, file access, and long runtime. Stage 1 uses synthetic needle-in-haystack data. Stage 2 adds curated recursive-editing tasks (key-value extraction, log parsing, transform-then-answer tasks). Risks include the agent failing to use tools correctly, getting stuck in loops, or overfitting to synthetic tasks, plus instability in Python execution and sparse rewards during training.

TECHNICAL APPROACH

 We implement an RLM as a wrapper around a base LLM with a Python REPL holding the external context. The model issues REPL actions (a1…aT) and a final answer y; reward is correctness minus small penalties for excessive steps or tokens. The objective is to maximize expected reward across tasks. We focus on recursive editing rather than evolutionary strategies per instructor guidance. Week 4’s milestone is reproducing the RLM-minimal loop: injecting context into the REPL, generating Python queries, and retrieving the correct result.

Training follows two stages: behavior cloning on successful rollouts, then RL-style optimization (e.g., reward-weighted updates) to encourage efficient tool use. We keep the action space restricted (regex search, slicing, chunk summarization) to maintain stability. Validation includes correctness on test tasks, efficiency metrics, and anti-cheating tests (varying needle formats and positions). We run locally with small models and limited episodes, logging runtime, tokens, and REPL steps to ensure reproducibility.

RESULTS

Our minimal RLM implementation successfully reproduces the RLM-minimal loop end-to-end. The deterministic baseline agent achieves 100% accuracy (10/10 episodes) on synthetic needle-in-haystack tasks, demonstrating that the REPL injection, code generation, and result retrieval pipeline works correctly.

Performance metrics from batch evaluation (10 episodes):
- Accuracy: 100.00% (all episodes correct)
- Average runtime: 0.000122 seconds per episode
- Average REPL steps: 1.00 step per episode
- REPL execution time: 0.000032-0.000347 seconds (mean: 0.000077 sec)

Edge case tests all pass:
- Missing needle (num_needles=0): Agent correctly returns "Needle not found"
- Multiple needles (num_needles=2): Agent retrieves correct value
- Long haystack (30 sentences, ~1771 characters): Agent successfully retrieves needle in < 0.001 seconds

Current limitations: The implementation uses deterministic regex search rather than LLM-generated actions, handles only single-step retrieval (no multi-step reasoning), and has no training loop. The safe execution environment adds minimal overhead (< 0.001 sec per REPL call), and we encountered no unexpected challenges in the baseline implementation. Resource usage is minimal: episodes complete in under 1 millisecond on average, making the system computationally efficient for validation.

SELF-CRITIQUE (Week 4)

OBSERVE

 Reviewing our Week 4 deliverable, our problem framing and minimal RLM environment are solid, and the notebook runs end-to-end. However, our implementation is still mostly a deterministic scaffold, and we have not yet connected an actual LLM or tested multi-step recursive behavior. Several components remain unvalidated beyond simple synthetic tasks.

ORIENT

 Strengths:
- Clear problem definition and working minimal RLM pipeline.
- Safe Python execution and reproducible batch results.
- Strong roadmap with measurable success metrics.

Areas for Improvement:
- No LLM-generated tool actions yet.
- Tasks are too simple to demonstrate real recursive editing.
- Training loop is only outlined, not implemented.

Critical Risks / Assumptions

 We assume our sandbox will behave correctly once the agent starts producing noisy or malformed Python, which is untested. We also assume synthetic tasks will transfer to more realistic ones when we introduce learning.

DECIDE

 Concrete Next Actions:
- Integrate a small local or API LLM to generate Python actions.
- Add tasks that require multi-step reasoning.
- Implement a basic imitation-learning loop using successful trajectories.

ACT

 Resource Needs
 We need example training scripts from rlm-minimal, access to a small model for testing, and guidance on reward shaping and safe sandbox extensions once the model begins producing real code.

