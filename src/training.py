"""
Training pipeline for RLM agents.

Provides:
- load_model: Load a causal LM with LoRA (fresh or from checkpoint)
- collect_trajectories: Generate training data using any agent
- BehaviorCloningTrainer: Phase 1 — SFT on successful heuristic trajectories
- ReinforceTrainer: Phase 2 — reward-weighted policy gradient (REINFORCE)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Callable
import os
import random

# Torch imports are deferred so the module is importable without GPU deps.
# They are imported at the top of functions that need them.
_torch_available = False
try:
    import torch
    import torch.nn.functional as F
    _torch_available = True
except ImportError:
    pass

from src.utils import (
    safe_execute_code,
    compute_reward,
    generate_task,
    generate_kv_extraction_task,
    generate_multistep_task,
)
from src.models import Agent, _LLM_SYSTEM_PROMPT, extract_code_from_response


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def set_training_seed(seed: int) -> None:
    """
    Set RNG seeds for reproducible trajectory sampling and trainer init.

    Use a nonnegative seed before collecting data or launching HF trainers.
    """
    if seed < 0:
        return
    random.seed(seed)
    if _torch_available:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


LORA_PRESETS = {
    "attention": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
    "mlp+head": ["gate_proj", "up_proj", "down_proj", "lm_head"],
    "attention+mlp": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "all": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    "head": ["lm_head"],
}


def load_model(
    model_id_or_path: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    load_in_4bit: bool = True,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_target_modules: Optional[List[str]] = None,
    lora_preset: Optional[str] = None,
    device_map: str = "auto",
    use_unsloth: bool = False,
):
    """
    Load a causal LM for training.

    If model_id_or_path points to a checkpoint directory (contains
    adapter_config.json), load saved LoRA adapters. Otherwise load the
    base model and attach fresh LoRA adapters.

    Args:
        lora_target_modules: Explicit list of module names.
        lora_preset: Shorthand — one of "attention", "mlp", "mlp+head",
                     "attention+mlp", "all", "head".  Overridden by
                     lora_target_modules if both are given.
        use_unsloth: If True, use unsloth FastLanguageModel for 2-5x
                     faster training with automatic bf16 handling.

    Returns:
        (model, tokenizer)
    """
    is_checkpoint = os.path.isdir(model_id_or_path) and os.path.exists(
        os.path.join(model_id_or_path, "adapter_config.json")
    )

    if is_checkpoint:
        return _load_checkpoint(model_id_or_path, load_in_4bit, device_map)

    # --- Resolve LoRA targets ---
    if lora_target_modules is None:
        preset_key = lora_preset or "attention"
        lora_target_modules = LORA_PRESETS.get(preset_key, LORA_PRESETS["attention"])
    print(f"LoRA targets: {lora_target_modules}")

    # --- Unsloth fast path ---
    if use_unsloth:
        return _load_with_unsloth(
            model_id_or_path, lora_r, lora_alpha, lora_target_modules,
            load_in_4bit,
        )

    # --- Standard transformers + peft path ---
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

    # modules_to_save ensures lm_head is fully trainable (not low-rank)
    # when it appears in the target list, since LoRA on a tied embedding
    # can be tricky.  For other modules plain LoRA is fine.
    modules_to_save = []
    lora_modules = list(lora_target_modules)
    if "lm_head" in lora_modules:
        lora_modules.remove("lm_head")
        modules_to_save.append("lm_head")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_modules if lora_modules else ["q_proj", "v_proj"],
        modules_to_save=modules_to_save if modules_to_save else None,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


def _load_with_unsloth(
    model_id: str,
    lora_r: int,
    lora_alpha: int,
    target_modules: List[str],
    load_in_4bit: bool,
):
    """Load model using unsloth for faster LoRA training."""
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        raise ImportError(
            "unsloth not installed. Install with:\n"
            "  pip install unsloth\n"
            "Then restart the runtime."
        )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=4096,
        load_in_4bit=load_in_4bit,
        dtype=None,  # auto-detect
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    model.print_trainable_parameters()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _load_checkpoint(
    checkpoint_dir: str,
    load_in_4bit: bool = True,
    device_map: str = "auto",
):
    """Load a model from a saved LoRA checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, PeftConfig

    peft_config = PeftConfig.from_pretrained(checkpoint_dir)
    base_model_id = peft_config.base_model_name_or_path

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(base_model, checkpoint_dir)

    # Ensure adapters are trainable for continued training
    for name, param in model.named_parameters():
        if "lora" in name.lower():
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Loaded checkpoint from {checkpoint_dir}")
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_prompt(question: str, context: str) -> List[Dict[str, str]]:
    """Format a task as chat messages for the model."""
    context_preview = context[:500]
    if len(context) > 500:
        context_preview += f"\n... ({len(context)} total characters)"

    return [
        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context preview (the full text is available as the variable CONTEXT):\n"
                f"---\n{context_preview}\n---\n\n"
                f"Question: {question}\n\n"
                f"Write Python code to answer this question. "
                f"The full context is in the variable CONTEXT. "
                f"Print only the final answer."
            ),
        },
    ]


def format_sft_example(
    messages: List[Dict[str, str]], code: str, tokenizer
) -> str:
    """Format a (prompt, code) pair as a full chat string for SFT."""
    full_messages = messages + [
        {"role": "assistant", "content": f"```python\n{code}\n```"},
    ]
    return tokenizer.apply_chat_template(full_messages, tokenize=False)


# ---------------------------------------------------------------------------
# Trajectory data
# ---------------------------------------------------------------------------

@dataclass
class RLTrajectory:
    """A trajectory with all info needed for RL training."""
    context: str
    question: str
    correct_answer: str
    predicted_answer: str
    code: str
    reward: float
    is_correct: bool
    num_steps: int
    messages: List[Dict[str, str]]


def collect_trajectories(
    agent: Agent,
    task_generators: List[Callable],
    num_episodes: int = 500,
) -> List[RLTrajectory]:
    """Collect trajectories using any agent (for behavior cloning data)."""
    trajectories = []

    for i in range(num_episodes):
        gen = random.choice(task_generators)
        context, question, correct_answer = gen()

        predicted, transcript = agent.run_episode(context, question, correct_answer)
        is_correct = predicted == correct_answer
        reward = compute_reward(
            is_correct=is_correct,
            num_steps=len(transcript),
            max_steps=agent.max_steps,
        )

        # Extract the last (successful) code from transcript
        codes = [e.get("code", "") for e in transcript if "code" in e]
        code = codes[-1] if codes else ""

        messages = format_prompt(question, context)

        trajectories.append(RLTrajectory(
            context=context,
            question=question,
            correct_answer=correct_answer,
            predicted_answer=predicted,
            code=code,
            reward=reward,
            is_correct=is_correct,
            num_steps=len(transcript),
            messages=messages,
        ))

        if (i + 1) % 100 == 0:
            successes = sum(1 for t in trajectories if t.is_correct)
            print(f"  Collected {i + 1}/{num_episodes} trajectories ({successes} successful)")

    return trajectories


# ---------------------------------------------------------------------------
# Phase 1: Behavior Cloning (SFT)
# ---------------------------------------------------------------------------

class BehaviorCloningTrainer:
    """
    Supervised fine-tuning on successful heuristic trajectories.

    Uses trl.SFTTrainer with LoRA for parameter-efficient training.
    This gives the model a warm-start policy before RL.
    """

    def __init__(
        self,
        model,
        tokenizer,
        output_dir: str = "./checkpoints/bc",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        max_length: int = 2048,
        gradient_accumulation_steps: int = 2,
        weight_decay: float = 0.0,
        seed: Optional[int] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.seed = seed

    def train(self, trajectories: List[RLTrajectory]) -> Dict[str, Any]:
        """Run behavior cloning on successful trajectories."""
        from trl import SFTConfig, SFTTrainer
        from datasets import Dataset as HFDataset

        # Filter to successful trajectories with code
        successful = [t for t in trajectories if t.is_correct and t.code]
        print(f"Training on {len(successful)}/{len(trajectories)} successful trajectories")

        if not successful:
            raise ValueError("No successful trajectories to train on")

        # Format as text dataset
        texts = []
        for traj in successful:
            text = format_sft_example(traj.messages, traj.code, self.tokenizer)
            texts.append(text)

        dataset = HFDataset.from_dict({"text": texts})

        # Detect GPU capability: use bf16 on Ampere+, fp16 on older (T4, etc.)
        use_bf16 = False
        use_fp16 = False
        if _torch_available and torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 8:
                use_bf16 = True
            else:
                use_fp16 = True

        seed_kwargs: Dict[str, Any] = {}
        if self.seed is not None:
            seed_kwargs["seed"] = self.seed
            seed_kwargs["data_seed"] = self.seed

        training_args = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            max_length=self.max_length,
            logging_steps=10,
            save_steps=100,
            save_total_limit=2,
            bf16=use_bf16,
            fp16=use_fp16,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            dataset_text_field="text",
            gradient_checkpointing=True,
            optim="adamw_torch",
            weight_decay=self.weight_decay,
            **seed_kwargs,
        )

        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
        )

        result = trainer.train()
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        return {
            "train_loss": result.training_loss,
            "num_examples": len(successful),
            "epochs": self.num_epochs,
            "output_dir": self.output_dir,
        }


# ---------------------------------------------------------------------------
# Phase 2: REINFORCE
# ---------------------------------------------------------------------------

class ReinforceTrainer:
    """
    REINFORCE with reward-weighted policy gradient.

    Each iteration:
        1. Generate a batch of trajectories with the current policy
        2. Execute code in sandbox, compute rewards
        3. Compute policy gradient: nabla J = E[A * nabla log pi(code|prompt)]
           where A = reward - baseline (variance reduction)
        4. Update model weights via gradient ascent

    The reward function is:
        R = C - lambda_s * (T / T_max) - lambda_t * N_tokens
    """

    def __init__(
        self,
        model,
        tokenizer,
        task_generators: List[Callable],
        output_dir: str = "./checkpoints/rl",
        batch_size: int = 8,
        learning_rate: float = 1e-5,
        max_new_tokens: int = 512,
        max_length: int = 2048,
        baseline_ema: float = 0.9,
        max_grad_norm: float = 1.0,
        advantage_clip: float = 2.0,
        sample_temperature: float = 0.7,
        sample_top_p: float = 0.9,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.task_generators = task_generators
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.baseline_ema = baseline_ema
        self.max_grad_norm = max_grad_norm
        self.advantage_clip = advantage_clip
        self.sample_temperature = sample_temperature
        self.sample_top_p = sample_top_p

        # Optimizer over LoRA parameters only
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        self.reward_baseline = 0.0
        self.history: List[Dict[str, float]] = []

    def _generate_code(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[str, "torch.Tensor"]:
        """Generate code from prompt. Returns (response_text, generated_token_ids)."""
        with torch.no_grad():
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(
                input_text, return_tensors="pt", truncation=True, max_length=self.max_length
            ).to(self.model.device)
            input_length = inputs["input_ids"].shape[1]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.sample_temperature,
                top_p=self.sample_top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            generated_ids = outputs[0, input_length:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text, generated_ids

    def _compute_log_probs(
        self, messages: List[Dict[str, str]], generated_ids: "torch.Tensor"
    ) -> "torch.Tensor":
        """
        Compute sum of log probabilities of generated tokens under current policy.

        This is the key quantity for REINFORCE:
            log pi(a|s) = sum_t log pi(token_t | prompt, token_1..t-1)
        """
        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=self.max_length
        ).to(self.model.device)
        prompt_length = inputs["input_ids"].shape[1]

        # Full sequence: prompt + generated tokens
        full_ids = torch.cat(
            [inputs["input_ids"], generated_ids.unsqueeze(0)], dim=1
        )

        # Forward pass WITH gradients
        outputs = self.model(input_ids=full_ids)
        logits = outputs.logits

        # Logits at position i predict token at position i+1.
        # We want predictions for the generated tokens (positions prompt_length onwards).
        # So we take logits at positions [prompt_length-1, ..., -2].
        gen_logits = logits[:, prompt_length - 1:-1, :]
        log_probs = F.log_softmax(gen_logits, dim=-1)

        # Gather log probs of the actually generated tokens
        token_log_probs = log_probs.gather(
            -1, generated_ids.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs.sum()

    def train_step(self) -> Dict[str, float]:
        """
        One REINFORCE iteration:
        1. Rollout batch with current policy (no grad)
        2. Execute code, get rewards
        3. Compute advantages = reward - baseline
        4. Policy gradient loss = -advantage * log_prob
        5. Backprop and update
        """
        self.model.train()

        # --- 1. Collect trajectories ---
        trajectories: List[RLTrajectory] = []
        rollout_data = []

        for _ in range(self.batch_size):
            gen = random.choice(self.task_generators)
            context, question, correct_answer = gen()
            messages = format_prompt(question, context)

            response_text, generated_ids = self._generate_code(messages)
            code = extract_code_from_response(response_text)

            if code is None:
                # Could not extract code — treat as failed trajectory
                continue

            # Execute in sandbox
            exec_result = safe_execute_code(
                code, custom_globals={"CONTEXT": context}
            )

            if exec_result.ok and exec_result.stdout and exec_result.stdout.strip():
                predicted = exec_result.stdout.strip().splitlines()[-1].strip()
            else:
                predicted = ""

            is_correct = predicted == correct_answer
            reward = compute_reward(is_correct=is_correct, num_steps=1, max_steps=5)

            trajectories.append(RLTrajectory(
                context=context,
                question=question,
                correct_answer=correct_answer,
                predicted_answer=predicted,
                code=code,
                reward=reward,
                is_correct=is_correct,
                num_steps=1,
                messages=messages,
            ))
            rollout_data.append((messages, generated_ids))

        if not trajectories:
            return {
                "accuracy": 0.0, "avg_reward": 0.0,
                "loss": 0.0, "baseline": self.reward_baseline,
                "batch_size": 0,
            }

        # --- 2. Compute advantages ---
        rewards = [t.reward for t in trajectories]
        avg_reward = sum(rewards) / len(rewards)
        self.reward_baseline = (
            self.baseline_ema * self.reward_baseline
            + (1 - self.baseline_ema) * avg_reward
        )
        advantages = [r - self.reward_baseline for r in rewards]
        adv_tensor = torch.tensor(
            advantages, device=self.model.device, dtype=torch.float32
        )
        adv_std = adv_tensor.std(unbiased=False).clamp_min(1e-6)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / adv_std
        adv_tensor = torch.clamp(adv_tensor, -self.advantage_clip, self.advantage_clip)
        advantages = adv_tensor.tolist()

        # --- 3. Policy gradient loss ---
        self.optimizer.zero_grad()
        total_loss = 0.0

        for (messages, gen_ids), advantage in zip(rollout_data, advantages):
            log_prob = self._compute_log_probs(messages, gen_ids)
            # REINFORCE: loss = -advantage * log_prob
            loss = -advantage * log_prob
            # Accumulate gradients (will average below)
            (loss / len(trajectories)).backward()
            total_loss += loss.item()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad],
            self.max_grad_norm,
        )

        # --- 4. Update ---
        self.optimizer.step()

        accuracy = sum(1 for t in trajectories if t.is_correct) / len(trajectories)
        stats = {
            "accuracy": accuracy,
            "avg_reward": avg_reward,
            "loss": total_loss / len(trajectories),
            "baseline": self.reward_baseline,
            "batch_size": len(trajectories),
        }
        self.history.append(stats)
        return stats

    def train(self, num_iterations: int = 20) -> List[Dict[str, float]]:
        """Run multiple REINFORCE iterations."""
        print(f"\n--- REINFORCE Training: {num_iterations} iterations, batch={self.batch_size} ---")

        for i in range(num_iterations):
            stats = self.train_step()
            print(
                f"  Iter {i + 1}/{num_iterations}: "
                f"acc={stats['accuracy']:.1%} | "
                f"reward={stats['avg_reward']:.3f} | "
                f"loss={stats['loss']:.4f} | "
                f"baseline={stats['baseline']:.3f}"
            )

            # Save checkpoint every 5 iterations
            if (i + 1) % 5 == 0:
                save_dir = os.path.join(self.output_dir, f"iter_{i + 1}")
                os.makedirs(save_dir, exist_ok=True)
                self.model.save_pretrained(save_dir)
                self.tokenizer.save_pretrained(save_dir)
                print(f"    Checkpoint saved to {save_dir}")

        # Save final model
        os.makedirs(self.output_dir, exist_ok=True)
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"--- Training Complete. Model saved to {self.output_dir} ---")

        return self.history
