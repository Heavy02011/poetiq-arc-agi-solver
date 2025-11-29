# Poetiq ARC-AGI Solver: Deep Code Review & Performance Analysis

## Executive Summary

This document provides a comprehensive deep code review of Poetiq's record-breaking ARC-AGI solver implementation. The solution achieves state-of-the-art performance on the ARC-AGI-1 and ARC-AGI-2 benchmarks through a combination of **program synthesis** (code generation), **iterative refinement with feedback**, **parallel expert ensemble**, and **intelligent voting/ranking mechanisms**.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Key Components Deep Dive](#key-components-deep-dive)
3. [Core Performance Techniques](#core-performance-techniques)
4. [Detailed Code Review](#detailed-code-review)
5. [Performance Optimization Strategies](#performance-optimization-strategies)
6. [Examples of Key Techniques](#examples-of-key-techniques)
7. [Summary of Key Ideas](#summary-of-key-ideas)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           main.py                                   │
│    (Loads ARC challenges, orchestrates solving, scores results)     │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          solve.py                                   │
│         (Entry point - calls solve_parallel_coding)                 │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    solve_parallel_coding.py                         │
│    (Runs multiple experts in parallel, aggregates via voting)       │
└───────┬─────────────────────────────────────────────┬───────────────┘
        │                                             │
        ▼                                             ▼
┌───────────────────┐                       ┌───────────────────┐
│   solve_coding    │ ... (N experts) ...   │   solve_coding    │
│   (Expert 1)      │                       │   (Expert N)      │
└───────┬───────────┘                       └───────┬───────────┘
        │                                           │
        ▼                                           ▼
┌───────────────────────────────────────────────────────────────────┐
│                           llm.py                                   │
│           (Async LLM calls with rate limiting & retry)             │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│                         sandbox.py                                 │
│        (Secure code execution in isolated subprocess)              │
└───────────────────────────────────────────────────────────────────┘
```

### Component Relationships

| Component | Responsibility |
|-----------|----------------|
| `main.py` | Entry point, loads data, coordinates evaluation, outputs results |
| `solve.py` | Simple wrapper that invokes the parallel coding solver |
| `solve_parallel_coding.py` | Orchestrates multiple experts, implements voting/ranking |
| `solve_coding.py` | Core iterative solver loop with feedback mechanism |
| `llm.py` | LLM API abstraction with rate limiting, retry logic, timeout handling |
| `sandbox.py` | Secure Python code execution in isolated subprocesses |
| `prompts.py` | System prompts for the LLM (3 solver variants + feedback prompt) |
| `config.py` | Expert configuration (model, parameters, voting settings) |
| `types.py` | TypedDict definitions for type safety |
| `scoring.py` | Score computation (allows 2 attempts per test) |
| `io.py` | Kaggle submission format conversion |
| `utils.py` | Utility for canonical output comparison |

---

## Key Components Deep Dive

### 1. Iterative Refinement Solver (`solve_coding.py`)

The heart of the system is an **iterative refinement loop** that:

1. **Generates Python code** to solve ARC transformation problems
2. **Executes code** in a sandbox against training examples
3. **Collects feedback** on failures (shape mismatches, value errors)
4. **Feeds feedback back** to the LLM to refine solutions

```python
# Core loop structure (simplified)
for iteration in range(max_iterations):
    # 1. Build prompt with problem description
    prompt = build_prompt(solver_prompt, problem=problem_str)
    
    # 2. Include feedback from previous attempts (key innovation!)
    if previous_solutions:
        selected = sample_with_probability(solutions, selection_probability)
        prompt += create_examples(selected, improving_order=True)
    
    # 3. Call LLM to generate code
    response = await llm(model, prompt, temperature=1.0, ...)
    code = parse_code_from_llm(response)
    
    # 4. Execute on training examples
    train_results, test_results = await eval_on_train_and_test(code, ...)
    
    # 5. If all training examples pass, return solution
    if all(r["success"] for r in train_results):
        return ARCAGIResult(train_results, test_results, iteration)
    
    # 6. Build feedback for next iteration
    feedback, score = build_feedback(train_results, train_in, train_out)
    solutions.append(ARCAGISolution(code, feedback, score))
```

#### Key Features:

- **Soft scoring**: Even failed attempts get partial credit for correct pixels
- **Improving order**: Best solutions presented last (recency bias optimization)
- **Example shuffling**: Randomizes training example order per iteration (diversity)
- **Selection probability**: Stochastically selects which past solutions to include

### 2. Parallel Expert Ensemble (`solve_parallel_coding.py`)

Multiple "experts" run **concurrently** with different random seeds:

```python
async def solve_parallel_coding(..., expert_configs):
    # Each expert gets a unique seed offset
    for i, cfg in enumerate(expert_configs):
        cfg["seed"] += i * cfg["max_iterations"]
    
    # Run all experts in parallel
    tasks = [solve_coding(..., config=cfg) for cfg in expert_configs]
    results = await asyncio.gather(*tasks)
    
    # Aggregate results via voting
    return vote_and_rank(results)
```

### 3. Sophisticated Voting Mechanism

The voting system groups solutions by their **canonical output** and ranks them:

```python
# Group results by identical test outputs
candidate_buckets = {}  # Solutions that pass all training
failure_buckets = {}    # Solutions that fail some training

for result in results:
    key = canonical_test_key(result["results"])  # Output fingerprint
    if all_training_passed(result):
        candidate_buckets[key].append(result)
    else:
        failure_buckets[key].append(result)

# Ranking priority:
# 1. Passers with most votes (diversity-first within ties)
# 2. Failures with most votes (sorted by soft_score)
# 3. Remaining passers
# 4. Remaining failures
```

#### Voting Configuration Options:

| Option | Description |
|--------|-------------|
| `use_new_voting` | Enable advanced voting algorithm |
| `count_failed_matches` | Count failures that match passer outputs |
| `iters_tiebreak` | Use iteration count as tiebreaker |
| `low_to_high_iters` | Prefer solutions found earlier |

### 4. LLM Interface with Robustness (`llm.py`)

```python
async def llm(model, message, temperature, request_timeout, ...):
    attempt = 1
    while attempt <= retries:
        await limiters[model].wait()  # Rate limiting per model
        
        try:
            response = await acompletion(
                model=model,
                messages=[{"role": "user", "content": message}],
                temperature=temperature,
                timeout=current_request_timeout,
                **props[model],  # Model-specific settings
            )
            return response, duration, remaining_time, remaining_timeouts
            
        except (RateLimitError, InternalServerError, ...):
            # Transient errors - don't count against retries
            await asyncio.sleep(RETRY_DELAY_SEC)
            continue
            
        except TimeoutError:
            # Decrement timeout budget
            max_remaining_timeouts -= 1
            if max_remaining_timeouts <= 0:
                raise RuntimeError("Exceeded timeouts allotted")
```

#### Model-Specific Configurations:

| Model | Special Settings |
|-------|------------------|
| OpenAI GPT-5/5.1 | `reasoning_effort: "high"` |
| Claude Sonnet/Haiku 4.5 | `thinking.budget_tokens: 32000` |
| Gemini 2.5 Pro | `thinking.budget_tokens: 16000` |
| Gemini 3 Pro | Default settings |

### 5. Secure Code Sandbox (`sandbox.py`)

Code generated by the LLM is executed in an **isolated subprocess**:

```python
async def run(code, input_grid, timeout_s=1.5):
    # Build self-contained script
    script = f"""
    {code}
    if __name__ == '__main__':
        import json, numpy as np, scipy
        data = json.load(stdin)
        res = transform(np.array(data['input']))
        print(json.dumps({{"ok": True, "result": res.tolist()}}))
    """
    
    # Run in subprocess with timeout
    proc = await asyncio.create_subprocess_exec(
        sys.executable, script_path,
        stdin=PIPE, stdout=PIPE, stderr=PIPE,
        env={"PYTHONHASHSEED": "0"}  # Deterministic hashing
    )
    
    stdout, stderr = await asyncio.wait_for(
        proc.communicate(input=json_input),
        timeout=timeout_s
    )
```

---

## Core Performance Techniques

### Technique 1: Program Synthesis via Code Generation

Instead of directly predicting outputs, the system generates **Python code** that implements the transformation. This approach:

- ✅ Enables **generalization** beyond training examples
- ✅ Provides **interpretable** solutions
- ✅ Leverages LLM's coding capabilities
- ✅ Allows **complex transformations** via NumPy/SciPy

**Example prompt structure:**
```
You are an expert in solving ARC tasks by writing Python code...

Example #1
Input: <Diagram>...</Diagram>
Output: <Diagram>...</Diagram>

Challenge #1
Input: <Diagram>...</Diagram>

Write a transform(grid: np.ndarray) -> np.ndarray function...
```

### Technique 2: Iterative Refinement with Rich Feedback

The system provides **detailed, structured feedback** on failed attempts:

```python
def _build_feedback(train_results, train_in, train_out):
    for i, result in enumerate(train_results):
        if result["success"]:
            feedback.append(f"Solves Example #{i+1} correctly.")
        else:
            # Shape mismatch feedback
            if pred.shape != truth.shape:
                msg = f"Shape mismatch: {pred.shape} vs {truth.shape}"
            
            # Value mismatch feedback with visualization
            else:
                diff_grid = array_diff(pred, truth)  # "1/2" for mismatches
                msg = f"Visualization:\n{diff_grid}\n"
                msg += f"Output accuracy: {score:.2f}"
            
            feedback.append(msg)
    
    return feedback, mean_score
```

**Feedback example:**
```
Solves Example #1 incorrectly.
Your code's output does not match the expected output.

Visualization:
1 1/2 3
4 5/6 7
8 9 10/0

Output accuracy: 0.67 (0 is worst, 1 is best)
```

### Technique 3: Parallel Expert Diversification

Running multiple experts with **different random seeds** increases solution diversity:

```python
NUM_EXPERTS = 8  # Poetiq(Gemini-3-c) configuration

# Each expert explores different:
# - Training example orderings (shuffle_examples=True)
# - Past solution selections (selection_probability=1.0)
# - Seed offsets (seed += iteration_offset)
```

### Technique 4: Intelligent Solution Ranking

The voting system implements a **multi-criteria ranking**:

1. **Vote count**: More experts producing same output → higher confidence
2. **Training success**: Passers ranked above failures
3. **Soft score**: Partial correctness for tiebreaking
4. **Iteration count**: Earlier solutions can be preferred
5. **Diversity-first**: One solution per output group before duplicates

### Technique 5: Two-Attempt Submission Strategy

The Kaggle format allows **two guesses per test input**:

```python
def build_kaggle_two_attempts(results, test_in):
    for j in range(num_tests):
        attempts = []
        for result in results:  # Iterate by ranking
            output = result["results"][j]["output"]
            if output and len(attempts) < 2:
                attempts.append(output)
        return {"attempt_1": attempts[0], "attempt_2": attempts[1]}
```

This effectively doubles the success rate by submitting the **top-2 ranked solutions**.

---

## Detailed Code Review

### Code Quality Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Architecture** | ⭐⭐⭐⭐⭐ | Clean separation of concerns, well-defined interfaces |
| **Type Safety** | ⭐⭐⭐⭐ | Good use of TypedDict, Literal types |
| **Error Handling** | ⭐⭐⭐⭐ | Comprehensive retry logic, graceful degradation |
| **Async Design** | ⭐⭐⭐⭐⭐ | Excellent use of asyncio for parallelism |
| **Documentation** | ⭐⭐⭐ | Inline comments present, could use more docstrings |
| **Testability** | ⭐⭐⭐ | Functions are testable but no test suite included |
| **Security** | ⭐⭐⭐⭐ | Sandboxed execution, but subprocess has some risks |

### Strengths

1. **Modular Design**: Each component has a single responsibility
2. **Configurable**: Expert behavior controlled via `ExpertConfig` TypedDict
3. **Robust LLM Handling**: Rate limiting, retries, timeout budgets
4. **Efficient Parallelism**: `asyncio.gather()` for concurrent expert execution
5. **Deterministic Sandbox**: `PYTHONHASHSEED=0` ensures reproducibility

### Potential Improvements

1. **Add comprehensive test suite** for unit and integration testing
2. **Implement caching** for LLM responses to reduce API costs
3. **Add logging framework** instead of print statements
4. **Consider process pool** for sandbox execution (currently creates new process per run)
5. **Add input validation** for grid dimensions and values

---

## Performance Optimization Strategies

### 1. Concurrency & Parallelism

```python
# All experts run concurrently
tasks = [asyncio.create_task(solve_coding(...)) for cfg in configs]
results = await asyncio.gather(*tasks)

# All problems evaluated concurrently  
tasks = [asyncio.create_task(_eval_task_data(id, task)) for id, task in items]
for coro in asyncio.as_completed(tasks):
    ...  # Process as each completes
```

### 2. Resource Management

```python
# Increase file descriptor limit for high concurrency
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, hard))
```

### 3. Rate Limiting per Model

```python
limiters = {
    "gemini/gemini-3-pro-preview": Limiter(1.0),  # 1 req/sec
    "gemini/gemini-2.5-pro": Limiter(2.0),        # 2 req/sec
    ...
}
```

### 4. Timeout Budget Management

```python
# Track remaining time/timeouts across iterations
max_remaining_time -= duration
max_remaining_timeouts -= 1

# Exit early if budget exhausted
if max_remaining_time <= 0:
    raise RuntimeError("Exceeded time allotted")
```

---

## Examples of Key Techniques

### Example 1: Problem Representation

An ARC problem with input/output pairs is converted to a text diagram:

```
Example #1
Input:
<Diagram>
0 0 1 0
0 1 1 1
0 0 1 0
</Diagram>

Output:
<Diagram>
0 0 2 0
0 2 2 2
0 0 2 0
</Diagram>
```

### Example 2: Generated Solution Code

The LLM generates a Python function:

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """
    Replace all 1s that form a connected plus shape with 2s.
    The plus shape consists of a center cell and its 4-connected neighbors.
    """
    result = grid.copy()
    rows, cols = grid.shape
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 1:
                # Check if this 1 is part of a plus pattern
                is_center = (
                    i > 0 and i < rows-1 and 
                    j > 0 and j < cols-1 and
                    grid[i-1, j] == 1 and
                    grid[i+1, j] == 1 and
                    grid[i, j-1] == 1 and
                    grid[i, j+1] == 1
                )
                if is_center or _is_arm_of_plus(grid, i, j):
                    result[i, j] = 2
    
    return result
```

### Example 3: Feedback on Failed Attempt

When a solution fails, the system provides detailed feedback:

```
Solves Example #1 incorrectly.

Your code's output does not match the expected output.

Below is a visualization of the 2D array your code produced:
Correctly predicted values are shown as-is while incorrectly 
predicted values are shown in the format 'prediction/correct':

0 0 1/2 0
0 1/2 1/2 1/2
0 0 1/2 0

Output accuracy: 0.75 (0 is worst, 1 is best)
```

### Example 4: Voting in Action

With 8 experts producing solutions:

| Expert | Output Key | Training Pass? |
|--------|------------|----------------|
| 1 | `[[2,0],[0,2]]` | ✓ |
| 2 | `[[2,0],[0,2]]` | ✓ |
| 3 | `[[2,0],[0,2]]` | ✓ |
| 4 | `[[1,0],[0,1]]` | ✓ |
| 5 | `[[2,0],[0,2]]` | ✗ |
| 6 | `[[3,0],[0,3]]` | ✗ |
| 7 | `[[3,0],[0,3]]` | ✗ |
| 8 | `[[2,0],[0,2]]` | ✗ |

**Ranking result:**
1. `[[2,0],[0,2]]` - 5 votes (3 passers + 2 matching failures)
2. `[[1,0],[0,1]]` - 1 vote (passer)
3. `[[3,0],[0,3]]` - 2 votes (failures)

---

## Summary of Key Ideas

### 🔑 Key Idea 1: Code as the Universal Representation

**Instead of directly predicting grids, generate Python code that implements transformations.**

This enables:
- Generalization to unseen test inputs
- Interpretable, debuggable solutions
- Complex multi-step transformations
- Leveraging LLM coding capabilities

### 🔑 Key Idea 2: Iterative Self-Improvement

**Use execution feedback to iteratively refine solutions across multiple attempts.**

The loop: Generate → Execute → Evaluate → Feedback → Regenerate

Key innovations:
- Soft scoring (partial credit)
- Visual diff feedback
- Best solutions presented last (recency bias)
- Stochastic selection of past attempts

### 🔑 Key Idea 3: Ensemble Diversity via Parallelism

**Run multiple experts with different random configurations simultaneously.**

Diversity sources:
- Different random seeds
- Different training example orderings
- Different solution selections for feedback
- Potentially different prompts (SOLVER_PROMPT_1/2/3)

### 🔑 Key Idea 4: Consensus-Based Ranking

**Trust solutions that multiple independent experts agree on.**

The voting mechanism:
- Groups solutions by identical outputs
- Prioritizes high-vote groups
- Uses soft scores as tiebreakers
- Ensures diversity in top selections

### 🔑 Key Idea 5: Robust API Handling

**Handle API failures gracefully to maximize problem coverage.**

Techniques:
- Per-model rate limiting
- Transient error retry (doesn't count against budget)
- Timeout budgets (fail gracefully, move on)
- Accumulated time/timeout tracking

### 🔑 Key Idea 6: Two-Attempt Exploitation

**Submit top-2 ranked solutions to maximize success rate.**

By voting and ranking all solutions, the system can provide a highly confident first attempt and a diverse second attempt.

---

## Conclusion

Poetiq's ARC-AGI solver achieves state-of-the-art performance through a sophisticated combination of:

1. **Program synthesis** - Generating code instead of direct predictions
2. **Iterative refinement** - Using execution feedback to improve solutions
3. **Parallel ensemble** - Running multiple experts for diversity
4. **Intelligent voting** - Aggregating solutions by consensus
5. **Robust engineering** - Handling API failures, timeouts, rate limits

The architecture is well-designed, modular, and extensible. The key insight is that **code generation + iterative refinement + ensemble voting** creates a powerful system that can discover and verify complex transformations that would be difficult to predict directly.

---

*Document generated as part of code review for the Poetiq ARC-AGI Solver repository.*
