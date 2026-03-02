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
