"""Exercise 4: generate test cases locally and use them to find known bugs.

Exercise summary
----------------
This example gives Ornith the specification and implementation of two small
Python functions. Each implementation contains a deliberate defect. The model
generates five data-driven test cases per function, and trusted Python code
executes those cases against both the buggy implementation and a reference
implementation.

The model returns test data rather than executable Python source. This is a
useful teaching and safety pattern: an LLM can propose inputs and expected
outputs, while the application retains control over the code that actually
runs. The generated cases are similar to parameters supplied to a pytest
``parametrize`` test.

Why this can save time and API cost
-----------------------------------
Developers often need many boundary and edge-case ideas while writing tests.
A local model can generate those candidates repeatedly without a hosted API
charge. Automated evaluation then rejects incorrect expectations and measures
whether the remaining cases expose the planted bug.

Steps in this example
---------------------
1. Import the required libraries.
2. Configure the local model connection.
3. Define buggy functions and trusted reference implementations.
4. Describe each test-generation task.
5. Build a task-specific JSON schema.
6. Ask Ornith for structured test cases.
7. Run each generated case through trusted Python functions.
8. Score correctness and bug-detection value.
9. Aggregate metrics across both tasks.
10. Print the complete benchmark payload as JSON.

Run with ``python ornith_unit_test_generation.py`` while LM Studio is serving
the Ornith model. Redirect stdout into ``artifacts`` to preserve the run.
"""

# Step 1: Import the libraries needed for API calls, timing, and JSON output.
from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from time import perf_counter
from typing import Any

from openai import OpenAI, OpenAIError


# Step 2: Configure the local LM Studio connection.
BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.getenv("LM_STUDIO_MODEL", "ornith-1.0-9b")
API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")


# Step 3: Define deliberately buggy functions and their trusted references.
#
# The buggy leap-year function misses the Gregorian century exception. For
# example, 1900 is divisible by 4 but is not a leap year.
def buggy_is_leap_year(year: int) -> bool:
    """Return the intentionally incomplete leap-year calculation."""

    return year % 4 == 0


def reference_is_leap_year(year: int) -> bool:
    """Return the correct Gregorian leap-year calculation."""

    return year % 400 == 0 or (year % 4 == 0 and year % 100 != 0)


# The buggy clamp subtracts one from the upper boundary. It therefore changes
# values that are exactly equal to the valid maximum.
def buggy_clamp(value: int, lower: int, upper: int) -> int:
    """Return the deliberately incorrect bounded value."""

    return max(lower, min(value, upper - 1))


def reference_clamp(value: int, lower: int, upper: int) -> int:
    """Return value constrained to the inclusive lower and upper boundaries."""

    return max(lower, min(value, upper))


# Step 4: Describe the two generation tasks in data.
#
# Functions cannot be represented in JSON, so each task stores callable Python
# objects separately from the strings sent to the model. This keeps execution
# trusted while preserving the exact specification and buggy source in the
# final artifact.
TEST_TASKS: list[dict[str, Any]] = [
    {
        "task_id": "leap_year",
        "function_name": "is_leap_year",
        "argument_names": ["year"],
        "expected_type": "boolean",
        "specification": (
            "Return True for Gregorian leap years. A year divisible by 4 is a "
            "leap year, except a year divisible by 100 is not, unless it is "
            "also divisible by 400."
        ),
        "buggy_source": "def is_leap_year(year):\n    return year % 4 == 0",
        "buggy_function": buggy_is_leap_year,
        "reference_function": reference_is_leap_year,
    },
    {
        "task_id": "inclusive_clamp",
        "function_name": "clamp",
        "argument_names": ["value", "lower", "upper"],
        "expected_type": "integer",
        "specification": (
            "Return value constrained to the inclusive interval from lower "
            "through upper. Assume lower is less than or equal to upper."
        ),
        "buggy_source": (
            "def clamp(value, lower, upper):\n"
            "    return max(lower, min(value, upper - 1))"
        ),
        "buggy_function": buggy_clamp,
        "reference_function": reference_clamp,
    },
]


def create_client() -> OpenAI:
    """Return an OpenAI-compatible client for the local model."""

    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


# Step 5: Build a JSON schema matching the task's arguments and return type.
def build_response_format(task: dict[str, Any]) -> dict[str, Any]:
    """Return the structured-output schema for one test-generation task."""

    argument_count = len(task["argument_names"])
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{task['task_id']}_test_cases",
            "schema": {
                "type": "object",
                "properties": {
                    "test_cases": {
                        "type": "array",
                        "minItems": 5,
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "args": {
                                    "type": "array",
                                    "items": {"type": "integer"},
                                    "minItems": argument_count,
                                    "maxItems": argument_count,
                                },
                                "expected": {"type": task["expected_type"]},
                                "reason": {"type": "string"},
                            },
                            "required": ["name", "args", "expected", "reason"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["test_cases"],
                "additionalProperties": False,
            },
        },
    }


# Step 6: Ask Ornith to propose exactly five structured test cases.
def generate_test_cases(
    client: OpenAI, task: dict[str, Any]
) -> dict[str, Any]:
    """Return model-generated cases with timing and token metadata."""

    prompt = (
        "Generate exactly five high-value unit test cases for the function. "
        "Include normal behavior, boundaries, and edge cases likely to reveal "
        "the defect. Arguments must follow this order: "
        f"{', '.join(task['argument_names'])}. Determine expected values from "
        "the specification, not from the buggy implementation.\n\n"
        f"Specification:\n{task['specification']}\n\n"
        f"Implementation under test:\n{task['buggy_source']}"
    )

    started = perf_counter()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You design precise, minimal software unit tests.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=build_response_format(task),
        temperature=0.2,
    )
    elapsed_seconds = perf_counter() - started

    content = completion.choices[0].message.content or "{}"
    parsed = json.loads(content)
    usage = completion.usage.model_dump() if completion.usage else None
    return {
        "test_cases": parsed.get("test_cases", []),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "usage": usage,
    }


# Step 7: Execute generated data through code controlled by this application.
def execute_case(
    test_case: dict[str, Any],
    buggy_function: Callable[..., Any],
    reference_function: Callable[..., Any],
) -> dict[str, Any]:
    """Evaluate one generated case without executing model-generated code."""

    args = test_case.get("args", [])
    expected = test_case.get("expected")

    try:
        reference_result = reference_function(*args)
        buggy_result = buggy_function(*args)
    except (TypeError, ValueError, OverflowError) as error:
        return {
            "runnable": False,
            "reference_result": None,
            "buggy_result": None,
            "expected_is_correct": False,
            "detects_bug": False,
            "error": str(error),
        }

    # Step 8: Score whether the model's expected value is correct and whether
    # the test exposes different behavior in the buggy implementation.
    expected_is_correct = expected == reference_result
    detects_bug = expected_is_correct and buggy_result != expected

    return {
        "runnable": True,
        "reference_result": reference_result,
        "buggy_result": buggy_result,
        "expected_is_correct": expected_is_correct,
        "detects_bug": detects_bug,
        "error": None,
    }


# Step 9: Generate, execute, and aggregate both benchmark tasks.
def run_benchmark() -> dict[str, Any]:
    """Return generated tests, evaluations, and summary metrics."""

    client = create_client()
    task_results: list[dict[str, Any]] = []

    for task in TEST_TASKS:
        public_task = {
            key: value
            for key, value in task.items()
            if key not in {"buggy_function", "reference_function"}
        }

        try:
            generated = generate_test_cases(client, task)
            evaluated_cases = []
            for test_case in generated["test_cases"]:
                evaluated_cases.append(
                    {
                        "generated_test": test_case,
                        "evaluation": execute_case(
                            test_case,
                            task["buggy_function"],
                            task["reference_function"],
                        ),
                    }
                )

            task_results.append(
                {
                    "task": public_task,
                    "cases": evaluated_cases,
                    "elapsed_seconds": generated["elapsed_seconds"],
                    "usage": generated["usage"],
                    "caught_planted_bug": any(
                        case["evaluation"]["detects_bug"]
                        for case in evaluated_cases
                    ),
                    "error": None,
                }
            )
        except (OpenAIError, json.JSONDecodeError, KeyError, TypeError) as error:
            task_results.append(
                {
                    "task": public_task,
                    "cases": [],
                    "elapsed_seconds": None,
                    "usage": None,
                    "caught_planted_bug": False,
                    "error": str(error),
                }
            )

    all_cases = [
        case for task_result in task_results for case in task_result["cases"]
    ]
    return {
        "exercise": "unit_test_generation_and_bug_detection",
        "backend": "local_lm_studio",
        "model": MODEL,
        "base_url": BASE_URL,
        "safety_note": (
            "The model generated test data only. This program did not execute "
            "arbitrary source code produced by the model."
        ),
        "tasks": task_results,
        "summary": {
            "total_tasks": len(task_results),
            "completed_tasks": sum(result["error"] is None for result in task_results),
            "tasks_that_caught_planted_bug": sum(
                result["caught_planted_bug"] for result in task_results
            ),
            "generated_cases": len(all_cases),
            "runnable_cases": sum(
                case["evaluation"]["runnable"] for case in all_cases
            ),
            "cases_with_correct_expected_value": sum(
                case["evaluation"]["expected_is_correct"] for case in all_cases
            ),
            "bug_detecting_cases": sum(
                case["evaluation"]["detects_bug"] for case in all_cases
            ),
            "elapsed_seconds": round(
                sum(result["elapsed_seconds"] or 0 for result in task_results), 3
            ),
            "total_tokens": sum(
                (result["usage"] or {}).get("total_tokens", 0)
                for result in task_results
            ),
            "marginal_hosted_api_cost_usd": 0,
        },
    }


# Step 10: Print the full result payload when the example runs directly.
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(json.dumps(run_benchmark(), indent=2, ensure_ascii=False))
