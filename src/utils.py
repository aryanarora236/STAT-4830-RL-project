"""
Utility functions for the RLM needle-in-haystack retrieval system.

This module provides:
- ExecResult: Dataclass for code execution results
- safe_execute_code: Secure sandboxed Python code execution
- generate_task: Synthetic needle-in-haystack task generator
"""

from dataclasses import dataclass
import io
import sys
import signal
import time
import contextlib
import random
from typing import Any, Dict, Optional


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
    'min', 'max', 'sum', 'abs', 'round', 'type', 'isinstance'
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
