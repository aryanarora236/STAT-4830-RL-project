"""
Utility functions for the RLM needle-in-haystack retrieval system.

This module provides:
- ExecResult: Dataclass for code execution results
- safe_execute_code: Secure sandboxed Python code execution
- generate_task: Synthetic needle-in-haystack task generator
- generate_multistep_task: Tasks requiring multi-step REPL reasoning
- generate_kv_extraction_task: Key-value extraction from structured logs
- compute_reward: Reward computation for RL training
"""

from dataclasses import dataclass
import io
import sys
import signal
import time
import contextlib
import random
import string
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExecResult:
    """Result of executing Python code in a sandboxed environment."""
    ok: bool
    stdout: str
    stderr: str
    runtime_sec: float


# Constants for safe execution
TIMEOUT_SECONDS = 5

_EXEC_WHITELISTED_BUILTINS = (
    'print', 'len', 'str', 'int', 'float', 'range', 'dict', 'list', 'set', 'tuple',
    'min', 'max', 'sum', 'abs', 'round', 'type', 'isinstance',
    'enumerate', 'sorted', 'reversed', 'any', 'all', 'zip', 'map', 'filter',
    'bool', 'hasattr', 'getattr', 'chr', 'ord', 'hex', 'bin', 'oct',
    'True', 'False', 'None',
)

_EXEC_DENYLISTED_IMPORTS = (
    'os', 'sys', 'subprocess', 'threading', 'multiprocessing', 'shutil',
    'inspect', 'gc', 'resource', 'signal', '__import__'
)


def _timeout_handler(signum, frame):
    """Signal handler that raises TimeoutError when code execution times out."""
    raise TimeoutError("Code execution timed out")


def safe_execute_code(code: str, custom_globals: Optional[Dict[str, Any]] = None) -> ExecResult:
    """
    Execute Python code in a secure sandboxed environment.
    
    Security features:
    - I/O capture (stdout/stderr) for result extraction
    - 5-second timeout to prevent infinite loops
    - Whitelisted built-ins only (print, len, str, etc.)
    - Blocked dangerous imports (os, sys, subprocess, etc.)
    - Custom __import__ handler to enforce restrictions
    
    Args:
        code: Python code string to execute
        custom_globals: Optional dictionary of custom global variables to inject
        
    Returns:
        ExecResult: Contains ok flag, stdout, stderr, and runtime
    """
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    runtime_start = time.time()

    # Set signal handler for timeout
    signal.signal(signal.SIGALRM, _timeout_handler)

    # Prepare a safe global environment
    # Handle __builtins__ being either a dict or a module (Python version compatibility)
    builtins_dict = {}
    if isinstance(__builtins__, dict):
        for name in _EXEC_WHITELISTED_BUILTINS:
            if name in __builtins__:
                builtins_dict[name] = __builtins__[name]
    else:
        for name in _EXEC_WHITELISTED_BUILTINS:
            if hasattr(__builtins__, name):
                builtins_dict[name] = getattr(__builtins__, name)
    
    _safe_globals = {'__builtins__': builtins_dict}

    # Custom __import__ to block denylisted modules
    # Get the original __import__ function safely
    if isinstance(__builtins__, dict):
        original_import_func = __builtins__['__import__']
    else:
        original_import_func = getattr(__builtins__, '__import__')
    
    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        """Custom import function that blocks dangerous modules."""
        if name in _EXEC_DENYLISTED_IMPORTS:
            raise ImportError(f"Module '{name}' is not allowed to be imported.")
        return original_import_func(name, globals, locals, fromlist, level)

    _safe_globals['__builtins__']['__import__'] = _safe_import

    if custom_globals:
        _safe_globals.update(custom_globals)

    try:
        # Set alarm for timeout
        signal.alarm(TIMEOUT_SECONDS)
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            exec(code, _safe_globals, _safe_globals)
        ok = True
        stderr = stderr_capture.getvalue()
        # Check if stderr contains error messages (some errors might write to stderr without raising)
        if stderr and ("SyntaxError" not in stderr and "ImportError" not in stderr and "TimeoutError" not in stderr and "Runtime Error" not in stderr):
            # If there's stderr but it's not one of our known error types, still mark as error
            ok = False
    except TimeoutError as e:
        ok = False
        stderr_capture.write(f"Execution Timeout: {e}\n")
    except ImportError as e:
        ok = False
        stderr_capture.write(f"Import Error: {e}\n")
    except Exception as e:
        ok = False
        stderr_capture.write(f"Runtime Error: {type(e).__name__}: {e}\n")
    finally:
        # Clear the alarm
        signal.alarm(0)
        runtime_end = time.time()

    return ExecResult(
        ok=ok,
        stdout=stdout_capture.getvalue(),
        stderr=stderr_capture.getvalue(),
        runtime_sec=runtime_end - runtime_start
    )


def generate_task(num_sentences: int = 10, needle_key: str = 'SECRET_CODE', num_needles: int = 1):
    """
    Generates a synthetic needle-in-a-haystack task.
    
    Creates a haystack of random filler sentences with an embedded KEY=VALUE pair
    (the "needle"). Supports edge cases like missing needles and multiple needles.
    
    Args:
        num_sentences: Number of filler sentences in the haystack
        needle_key: The key for the KEY=VALUE needle pair
        num_needles: Number of times the needle should be embedded (0 for no needle)
        
    Returns:
        tuple: (haystack_str, question, correct_answer)
            - haystack_str: The generated context with or without the needle
            - question: The question to ask the agent
            - correct_answer: The expected answer for the needle
    """
    filler_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Never underestimate the power of a good book.",
        "The early bird catches the worm, or so they say.",
        "Technology has revolutionized the way we live and work.",
        "The serene mountains offered a perfect escape from city life.",
        "Learning new skills can open up many opportunities.",
        "Artificial intelligence is rapidly advancing its capabilities.",
        "The ocean depths hold countless mysteries yet to be discovered.",
        "A healthy diet and regular exercise are crucial for well-being.",
        "Creativity often flourishes in unexpected moments of inspiration.",
        "The historic monument stood tall, telling tales of a bygone era.",
        "Digital transformation is a continuous process for businesses.",
        "Effective communication is key to successful collaboration.",
        "The intricate patterns of nature always amaze scientists.",
        "Sustainable practices are essential for our planet's future."
    ]

    # Generate a random KEY=VALUE pair to serve as the needle
    needle_value = f'XYZ{random.randint(10000, 99999)}ABC'
    needle = f"{needle_key}={needle_value}"

    # Allow repetition if num_sentences > available sentences
    if num_sentences <= len(filler_sentences):
        haystack_list = random.sample(filler_sentences, k=num_sentences)
    else:
        # Repeat sentences to reach desired count
        haystack_list = []
        while len(haystack_list) < num_sentences:
            remaining = num_sentences - len(haystack_list)
            haystack_list.extend(random.sample(filler_sentences, k=min(len(filler_sentences), remaining)))

    # Embed the generated needle(s) into random positions
    correct_answer = "N/A"
    if num_needles > 0:
        correct_answer = needle_value
        for _ in range(num_needles):
            insert_position = random.randint(0, len(haystack_list))
            haystack_list.insert(insert_position, needle)
    elif num_needles == 0:
        correct_answer = "Needle not found"  # Explicitly state if no needle

    # Construct the full haystack string from the sentences
    haystack_str = ' '.join(haystack_list)

    # Formulate a clear question
    question = f"What is the value of {needle_key}?"

    return haystack_str, question, correct_answer


# ---------------------------------------------------------------------------
# Multi-step task generators (Week 6+)
# ---------------------------------------------------------------------------

def _random_id(length: int = 6) -> str:
    """Generate a random alphanumeric ID."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


def generate_kv_extraction_task(
    num_entries: int = 20,
    target_field: str = "status",
    filter_key: str = "level",
    filter_value: str = "ERROR",
) -> Tuple[str, str, str]:
    """
    Generate a structured log where the agent must filter rows by one field
    and then extract a value from a different field.

    Requires two REPL steps: (1) filter lines, (2) extract target value.

    Returns:
        (context, question, correct_answer)
    """
    levels = ["INFO", "DEBUG", "WARN", "ERROR"]
    statuses = ["pending", "active", "resolved", "timeout", "retry"]
    lines: List[str] = []
    answer_candidates: List[str] = []

    for _ in range(num_entries):
        lvl = random.choice(levels)
        rid = _random_id()
        status = random.choice(statuses)
        ts = f"2026-02-{random.randint(1, 28):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}"
        line = f"[{ts}] id={rid} level={lvl} status={status} msg=process_event"
        lines.append(line)
        if lvl == filter_value:
            answer_candidates.append(status)

    # Guarantee at least one matching entry
    if not answer_candidates:
        rid = _random_id()
        status = random.choice(statuses)
        ts = "2026-02-15T12:00"
        line = f"[{ts}] id={rid} level={filter_value} status={status} msg=process_event"
        pos = random.randint(0, len(lines))
        lines.insert(pos, line)
        answer_candidates.append(status)

    random.shuffle(lines)
    context = "\n".join(lines)

    # Ask for the status of the *first* matching ERROR entry (reading top-to-bottom)
    # Recompute to match the shuffled order
    first_match_status = None
    for line in lines:
        if f"level={filter_value}" in line:
            for part in line.split():
                if part.startswith(f"{target_field}="):
                    first_match_status = part.split("=", 1)[1]
                    break
            if first_match_status:
                break

    question = (
        f"In the log below, find the first entry where {filter_key}={filter_value} "
        f"and return its {target_field} value."
    )
    correct_answer = first_match_status or "not_found"
    return context, question, correct_answer


def generate_multistep_task(
    num_sentences: int = 15,
    num_keys: int = 3,
) -> Tuple[str, str, str]:
    """
    Generate a task requiring multiple REPL steps:
    1. Find all KEY=VALUE pairs in the haystack
    2. Sum their numeric suffixes
    3. Return the sum

    Returns:
        (context, question, correct_answer)
    """
    filler = [
        "Data processing completed for batch alpha.",
        "Checkpoint saved at iteration 500.",
        "Memory usage within normal parameters.",
        "Network latency measured at 42ms.",
        "Cache hit ratio improved to 87 percent.",
        "Background job scheduled for next cycle.",
        "Validation accuracy plateaued at current rate.",
        "Gradient norm stable across last 100 steps.",
        "Disk I/O throughput meets expected levels.",
        "Worker thread pool at full capacity.",
    ]

    keys = [f"METRIC_{chr(65 + i)}" for i in range(num_keys)]  # METRIC_A, METRIC_B, ...
    values = [random.randint(10, 99) for _ in range(num_keys)]
    needles = [f"{k}={v}" for k, v in zip(keys, values)]

    total = sum(values)

    if num_sentences <= len(filler):
        haystack_list = random.sample(filler, k=num_sentences)
    else:
        haystack_list = []
        while len(haystack_list) < num_sentences:
            remaining = num_sentences - len(haystack_list)
            haystack_list.extend(random.sample(filler, k=min(len(filler), remaining)))

    for needle in needles:
        pos = random.randint(0, len(haystack_list))
        haystack_list.insert(pos, needle)

    context = " ".join(haystack_list)
    question = (
        f"Find all METRIC_* keys in the text, extract their numeric values, "
        f"and return their sum."
    )
    correct_answer = str(total)
    return context, question, correct_answer


# ---------------------------------------------------------------------------
# Reward computation (Week 7+)
# ---------------------------------------------------------------------------

def compute_reward(
    is_correct: bool,
    num_steps: int,
    max_steps: int = 10,
    step_penalty: float = 0.05,
    token_count: int = 0,
    token_penalty: float = 0.0001,
) -> float:
    """
    Compute scalar reward for an episode.

    R = C - lambda_s * (steps / max_steps) - lambda_t * token_count

    Args:
        is_correct: Whether the final answer matched ground truth
        num_steps: Number of REPL steps taken
        max_steps: Maximum allowed steps (for normalisation)
        step_penalty: Weight for step cost
        token_count: Number of tokens used (0 if not tracked)
        token_penalty: Weight for token cost

    Returns:
        Scalar reward in roughly [-1, 1]
    """
    correctness = 1.0 if is_correct else 0.0
    step_cost = step_penalty * (num_steps / max(max_steps, 1))
    token_cost = token_penalty * token_count
    return correctness - step_cost - token_cost
