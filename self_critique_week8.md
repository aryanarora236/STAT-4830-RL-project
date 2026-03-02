# Self-Critique — Week 8 (Mar 3)

## OBSERVE

Reviewing our Week 8 state ahead of the lightning talk. Since Week 6, we have implemented the reward function, a training loop skeleton that collects and filters trajectories, and a side-by-side evaluation of both agents. However, the fundamental gap remains: no real LLM generates the REPL code. The heuristic agent demonstrates that multi-step reasoning works in principle, but the interesting research question — can a model learn to generate effective tool-use code? — is untouched.

## ORIENT

### Strengths (3 points)
- Complete pipeline from task generation → agent execution → reward computation → trajectory collection.
- Two working agents covering 3 task types with 100% accuracy on all tests (7/7 pass).
- Training loop collects batches and filters successful trajectories, ready for behavior cloning.

### Areas for Improvement (3 points)
- **LLM integration is overdue.** We have postponed this for four weeks. This is our biggest gap.
- **Training loop is a skeleton** — it collects data but does not update any model parameters.
- **Report and slides are behind schedule.** We need to catch up on versioned deliverables.

### Critical Risks / Assumptions
The biggest risk is that integrating an LLM will surface new problems (malformed code, sandbox failures, high latency) that we haven't encountered with template agents. We are also uncertain whether imitation learning from the heuristic agent's trajectories will produce a useful LLM policy, since the heuristic actions are quite formulaic.

## DECIDE

### Concrete Next Actions (3 points)
- **Over spring break:** Integrate a small model (Qwen-0.5B or an API like OpenAI) to generate Python REPL code for at least the needle task.
- **Week 9:** Run behavior cloning on successful heuristic trajectories and measure whether the LLM can replicate the heuristic strategy.
- **Week 10:** Add more diverse tasks and begin reward-weighted training.

## ACT

### Resource Needs
We need GPU access (Google Colab free tier may suffice for Qwen-0.5B) or API credits for a hosted model. We also need to study the rlm-minimal training loop implementation to understand their approach to behavior cloning with code-generating models.
