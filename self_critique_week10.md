# Self-Critique — Week 10 (Mar 27)

## OBSERVE

We pivoted the application domain from synthetic tasks (needle-in-haystack, KV extraction, aggregation) to No-Limit Texas Hold'em poker. The synthetic tasks validated the RLM framework but were too simple to demonstrate the value of long-context reasoning — the heuristic solved everything at 100% accuracy with rigid regex rules, and there was no meaningful long context to parse. Poker fixes this: optimal decisions depend on reading multi-hand opponent histories, computing stats like VPIP and aggression, and adjusting strategy accordingly. The framework (LLM + Python REPL sandbox) carries over directly.

We built a full poker environment with hand evaluation, a heuristic bot that follows a 3-step retrieve → compute → decide flow with opponent modeling, and a task generator that produces structured hand histories from consistent opponent archetypes.

## ORIENT

### Strengths
- The poker domain is a much better fit for the RLM thesis. Long context (hand history) genuinely matters for decision quality, unlike the synthetic tasks where context was just noise.
- The heuristic bot's 3-step reasoning flow (retrieve → compute → decide) gives us rich trajectory data for behavior cloning — the model doesn't just learn "what to do" but "how to reason about it."
- Five opponent archetypes with distinct profiles create diverse scenarios. The heuristic makes six types of opponent-adjusted decisions, showing that history parsing changes the final action.

### Areas for Improvement
- Opponent adjustments only fire ~9% of the time. For BC training data, we may need to bias toward scenarios where history matters (e.g., oversample hands against extreme opponents).
- We have not yet run the LLM on poker tasks, even zero-shot. We need a baseline number for how a model performs without training.
- The EV-based reward for RL is not implemented. We have the correctness-based reward from the original framework but need a poker-specific reward that measures whether the action was +EV.

### Critical Risks
The biggest risk is that behavior cloning produces a model that copies the heuristic's base strategy but ignores the opponent-modeling step, since adjustments are relatively rare in the training data. We need to ensure the training distribution emphasizes cases where history changes the decision. A secondary risk is that RL training against the heuristic may overfit to exploiting one specific opponent model rather than learning general exploitation skills.

## DECIDE

### Concrete Next Actions
- **This week**: Run zero-shot LLM evaluation on poker tasks to establish a pre-training baseline. Collect 500+ trajectories from the heuristic with reasoning traces.
- **Week 11**: Begin behavior cloning on Colab GPU. Measure if the BC model can replicate the heuristic's decisions, especially the opponent-adjusted ones.
- **Week 12**: Implement EV-based reward and begin REINFORCE training. The model plays against the heuristic bot, reward = chips won/lost.

## ACT

### Resource Needs
- GPU access (Google Colab) for fine-tuning Qwen 1.5B with LoRA
- Need to increase opponent adjustment rate in training data — either by oversampling or lowering thresholds further
- Should implement the EV-based reward before Week 12 so RL training can start on schedule
