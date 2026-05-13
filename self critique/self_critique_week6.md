# Self-Critique — Week 6 (Feb 20)

## OBSERVE

Reading our report and code critically after two more weeks of development. The Week 4 deterministic baseline remains solid. We have added multi-step task types and a heuristic agent, but the core loop still does not involve an actual LLM generating code. The mathematical formulation is cleaner now with an explicit reward function, but we have not yet computed rewards in practice.

## ORIENT

### Strengths (3 points)
- Modular agent architecture makes it easy to add new agents and task types.
- Multi-step tasks (KV extraction, aggregation) demonstrate the system can handle 2+ REPL steps.
- Safe execution sandbox continues to work reliably with no security issues.

### Areas for Improvement (3 points)
- We still lack LLM integration — the agents use hardcoded templates, not generated code.
- Our task diversity is limited to 3 synthetic types; real-world tasks would better demonstrate value.
- No training loop or reward computation in the codebase yet.

### Critical Risks / Assumptions
We assume the heuristic agent strategies will transfer to LLM-generated code, which is not guaranteed — LLMs may produce syntactically different code that our sandbox hasn't been tested against. We also assume the reward function weights (λ_s = 0.05, λ_t = 0.0001) are reasonable, but we have not validated them empirically.

## DECIDE

### Concrete Next Actions (3 points)
- Implement the reward function and integrate it into the evaluation framework.
- Build a training loop skeleton that collects trajectories and filters successful ones.
- Prepare a lightning talk summarising progress and approach for Week 8.

## ACT

### Resource Needs
We need to decide on an LLM for integration — options include a small HuggingFace model (runs locally but slow) or an API model (faster but costs money). We should also look at the rlm-minimal repo's training code for reference on behavior cloning implementation.
