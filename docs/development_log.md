Week 4 Development Log (Team)

Day 1
We met to understand the Week 4 deliverable requirements. After reading the instruction document, we realized the core requirement is having a minimal runnable optimization loop, not necessarily a trained agent. We discussed possible directions from the professor’s idea list and tried to decide between evolutionary strategies and recursive editing. After seeing the professor’s explicit note (“evo strategies are slow, do the other thing”), we unanimously shifted to recursive editing and RLM-style tool use. We also reviewed the RLM-minimal GitHub repo and skimmed the associated blog post.

Day 2
We spent time understanding the algorithm inside RLM-minimal. We summarized the main takeaway: the model uses a Python REPL as external memory rather than fitting the long context into the attention window. We discussed how to reproduce this using a toy needle-in-haystack task. We experimented with how to construct long random contexts and how to embed a needle string inside them. Some early attempts resulted in Python execution errors because we didn’t yet have a safe sandbox or restricted builtins.

Day 3
We started implementing the safe_exec function. Our first version failed because Python code could still import modules accidentally. We added explicit checks to block imports and replaced builtins with a restricted safe dictionary. Our execution wrapper now captures stdout and returns error messages without crashing the notebook. We confirmed that we could evaluate a small piece of code inside the REPL environment successfully.

Day 4
We built the initial needle task generator. We tested it with small context sizes first (500–2000 characters). We verified the needle insertion logic and confirmed that Python regex could reliably extract the value using the REPL. We wrote the first version of the deterministic “tool agent” that uses regex to find the needle. This allowed us to run the first working episode end-to-end. The predicted answer matched the ground truth.

Day 5
We expanded our test to longer contexts (10,000–30,000 characters). We added timing to episodes and saw that performance was stable. We added transcript logging so that each REPL call is recorded with code, stdout, errors, and runtime. This produced a clean example transcript for our report. We built a batch runner to compute accuracy and average runtime over multiple episodes. The deterministic agent solved every case, which is expected since it uses direct regex.

Day 6
We tried integrating a small HuggingFace model to generate Python code but ran into slow environment setup and model loading inconsistencies. Since Week 4 only requires a working minimal implementation, we postponed LLM generation to next week. We instead created a stub function that simulates what an LLM call would look like so the notebook structure is complete.

Day 7
We produced the notebook structure with sections for problem setup, mathematical formulation, implementation, validation, and next steps. We reran all cells start-to-finish to confirm nothing breaks. We collected metrics and example transcripts for the report. As a team, we discussed next steps: integrating real LLM code generation, reproducing more of the RLM-minimal pipeline, and extending the task family beyond simple needle retrieval. Everything needed for the Week 4 deliverable now exists in working form.
