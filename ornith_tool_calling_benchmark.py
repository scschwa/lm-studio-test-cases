"""Exercise 5: measure tool-selection and argument accuracy with Ornith.

Exercise summary
----------------
Tool calling lets a language model request a function that the surrounding
application knows how to execute. The model does not run Python code itself.
It returns a structured request containing a tool name and arguments; trusted
application code validates that request, calls an approved function, and can
then return the result to the user or model.

This benchmark gives Ornith three harmless local tools and five prompts with
known expected behavior. It measures whether the model selects the correct
tool, supplies exact arguments, handles a request for two tools, and avoids
calling a tool when a normal text answer is appropriate. Correctly requested
tools are executed against deterministic local stub functions.

Why this can save time and API cost
-----------------------------------
Tool-routing prompts often run many times during agent development and test
suites. A capable local model can exercise orchestration code without a paid
request for every test. Exact expected calls also provide a clear regression
metric rather than relying on subjective answer quality.

Steps in this example
---------------------
1. Import the required libraries.
2. Configure the local model connection.
3. Define safe Python functions that the application may execute.
4. Describe those functions to the model using tool schemas.
5. Define prompts and their expected tool calls.
6. Ask Ornith to choose tools for one prompt.
7. Parse and normalize the returned tool requests.
8. Compare requests with expected names and arguments.
9. Execute valid calls and aggregate benchmark metrics.
10. Print the complete result payload as JSON.

Run with ``python ornith_tool_calling_benchmark.py`` while LM Studio is serving
Ornith. Redirect stdout into ``artifacts`` to preserve the run.
"""

# Step 1: Import libraries for API calls, JSON parsing, timing, and typing.
from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from time import perf_counter
from typing import Any

from openai import OpenAI, OpenAIError


# Step 2: Configure the OpenAI-compatible LM Studio endpoint.
BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.getenv("LM_STUDIO_MODEL", "ornith-1.0-9b")
API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")


# Step 3: Define safe local functions that Python, not the model, executes.
#
# These deterministic stubs avoid filesystem, network, or production changes.
# They are sufficient to demonstrate how an application dispatches approved
# tool requests after receiving them from a model.
def lookup_issue(issue_id: str) -> dict[str, Any]:
    """Return synthetic issue data for a known identifier."""

    issues = {
        "AUTH-101": {"status": "open", "owner": "identity-team"},
        "PAY-201": {"status": "investigating", "owner": "payments-team"},
    }
    return issues.get(issue_id, {"status": "not_found", "owner": None})


def run_test_file(path: str) -> dict[str, Any]:
    """Return a simulated result without running a real test process."""

    return {"path": path, "status": "passed", "tests_run": 12}


def calculate_shipping(weight_kg: float, zone: int) -> dict[str, Any]:
    """Return a deterministic synthetic shipping calculation."""

    cost = round(4.50 + weight_kg * 0.80 + zone * 1.25, 2)
    return {"currency": "USD", "cost": cost}


# The dispatch table is an allowlist. A model can request only a named function
# present here; it cannot choose an arbitrary Python function to execute.
TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "lookup_issue": lookup_issue,
    "run_test_file": run_test_file,
    "calculate_shipping": calculate_shipping,
}


# Step 4: Describe the allowed functions to the model with JSON schemas.
#
# These descriptions guide tool selection, while each parameter schema limits
# the shape and type of arguments the model should return.
TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "lookup_issue",
            "description": "Look up a software issue by its issue identifier.",
            "parameters": {
                "type": "object",
                "properties": {"issue_id": {"type": "string"}},
                "required": ["issue_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_test_file",
            "description": "Run one test file identified by its repository path.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_shipping",
            "description": "Calculate shipping cost from weight and shipping zone.",
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_kg": {"type": "number"},
                    "zone": {"type": "integer"},
                },
                "required": ["weight_kg", "zone"],
                "additionalProperties": False,
            },
        },
    },
]


# Step 5: Define benchmark prompts and exact expected tool calls.
#
# The last case requires no tool. It tests whether the model can answer normally
# instead of treating every prompt as a reason to call a function.
BENCHMARK_CASES = [
    {
        "case_id": "single_issue_lookup",
        "prompt": "Look up issue AUTH-101.",
        "expected_calls": [
            {"name": "lookup_issue", "arguments": {"issue_id": "AUTH-101"}}
        ],
    },
    {
        "case_id": "single_test_run",
        "prompt": "Run the test file tests/test_checkout.py.",
        "expected_calls": [
            {"name": "run_test_file", "arguments": {"path": "tests/test_checkout.py"}}
        ],
    },
    {
        "case_id": "shipping_calculation",
        "prompt": "Calculate shipping for 12.5 kilograms going to zone 3.",
        "expected_calls": [
            {
                "name": "calculate_shipping",
                "arguments": {"weight_kg": 12.5, "zone": 3},
            }
        ],
    },
    {
        "case_id": "multiple_tools",
        "prompt": (
            "Look up issue PAY-201 and run tests/test_payments.py. Use both "
            "available tools needed for the request."
        ),
        "expected_calls": [
            {"name": "lookup_issue", "arguments": {"issue_id": "PAY-201"}},
            {
                "name": "run_test_file",
                "arguments": {"path": "tests/test_payments.py"},
            },
        ],
    },
    {
        "case_id": "no_tool_needed",
        "prompt": "Explain what a regression test is in one short sentence.",
        "expected_calls": [],
    },
]


def create_client() -> OpenAI:
    """Return a client connected to LM Studio."""

    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


# Step 6: Ask Ornith to decide whether one or more tools are appropriate.
def request_tool_selection(client: OpenAI, prompt: str) -> dict[str, Any]:
    """Return the model message plus request metrics for one prompt."""

    started = perf_counter()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Use the provided tools when they are needed. Request every "
                    "tool needed by the user. Answer normally when no tool fits."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        tools=TOOLS,
        tool_choice="auto",
        temperature=0,
    )
    elapsed_seconds = perf_counter() - started
    message = completion.choices[0].message

    return {
        "content": message.content,
        "tool_calls": message.tool_calls or [],
        "finish_reason": completion.choices[0].finish_reason,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "usage": completion.usage.model_dump() if completion.usage else None,
    }


# Step 7: Parse SDK tool-call objects into plain JSON-serializable dictionaries.
def normalize_tool_calls(tool_calls: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Return parsed calls and any argument-parsing errors."""

    normalized: list[dict[str, Any]] = []
    errors: list[str] = []

    for tool_call in tool_calls:
        try:
            arguments = json.loads(tool_call.function.arguments)
            normalized.append(
                {"name": tool_call.function.name, "arguments": arguments}
            )
        except (AttributeError, json.JSONDecodeError, TypeError) as error:
            errors.append(str(error))

    return normalized, errors


def canonical_calls(calls: list[dict[str, Any]]) -> list[str]:
    """Return order-independent strings for exact call comparison."""

    return sorted(json.dumps(call, sort_keys=True) for call in calls)


# Step 8: Compare actual requests with exact expected names and arguments.
def evaluate_calls(
    expected_calls: list[dict[str, Any]], actual_calls: list[dict[str, Any]]
) -> dict[str, bool]:
    """Return tool-selection and full exact-match scores."""

    expected_names = sorted(call["name"] for call in expected_calls)
    actual_names = sorted(call["name"] for call in actual_calls)
    return {
        "tool_names_correct": actual_names == expected_names,
        "exact_calls_correct": canonical_calls(actual_calls)
        == canonical_calls(expected_calls),
    }


# Step 9: Execute allowlisted calls and aggregate benchmark results.
def execute_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Execute valid requested tools through the explicit allowlist."""

    executions: list[dict[str, Any]] = []
    for call in calls:
        function = TOOL_FUNCTIONS.get(call["name"])
        if function is None:
            executions.append({"call": call, "result": None, "error": "not allowed"})
            continue

        try:
            result = function(**call["arguments"])
            executions.append({"call": call, "result": result, "error": None})
        except (TypeError, ValueError) as error:
            executions.append({"call": call, "result": None, "error": str(error)})

    return executions


def run_benchmark() -> dict[str, Any]:
    """Run all tool-choice cases and return detailed accuracy metrics."""

    client = create_client()
    results: list[dict[str, Any]] = []

    for case in BENCHMARK_CASES:
        try:
            model_run = request_tool_selection(client, case["prompt"])
            actual_calls, parse_errors = normalize_tool_calls(model_run["tool_calls"])
            evaluation = evaluate_calls(case["expected_calls"], actual_calls)
            results.append(
                {
                    "case": case,
                    "model_content": model_run["content"],
                    "actual_calls": actual_calls,
                    "parse_errors": parse_errors,
                    "evaluation": evaluation,
                    "executions": execute_calls(actual_calls),
                    "finish_reason": model_run["finish_reason"],
                    "elapsed_seconds": model_run["elapsed_seconds"],
                    "usage": model_run["usage"],
                    "error": None,
                }
            )
        except OpenAIError as error:
            results.append(
                {
                    "case": case,
                    "model_content": None,
                    "actual_calls": [],
                    "parse_errors": [],
                    "evaluation": {
                        "tool_names_correct": False,
                        "exact_calls_correct": False,
                    },
                    "executions": [],
                    "finish_reason": None,
                    "elapsed_seconds": None,
                    "usage": None,
                    "error": str(error),
                }
            )

    return {
        "exercise": "tool_calling_accuracy",
        "backend": "local_lm_studio",
        "model": MODEL,
        "base_url": BASE_URL,
        "security_note": (
            "The model requested tools; Python executed only functions in the "
            "explicit TOOL_FUNCTIONS allowlist."
        ),
        "cases": results,
        "summary": {
            "total_cases": len(results),
            "completed_cases": sum(result["error"] is None for result in results),
            "tool_name_correct_cases": sum(
                result["evaluation"]["tool_names_correct"] for result in results
            ),
            "exact_call_correct_cases": sum(
                result["evaluation"]["exact_calls_correct"] for result in results
            ),
            "parse_error_count": sum(
                len(result["parse_errors"]) for result in results
            ),
            "executed_tool_calls": sum(
                len(result["executions"]) for result in results
            ),
            "elapsed_seconds": round(
                sum(result["elapsed_seconds"] or 0 for result in results), 3
            ),
            "total_tokens": sum(
                (result["usage"] or {}).get("total_tokens", 0) for result in results
            ),
            "marginal_hosted_api_cost_usd": 0,
        },
    }


# Step 10: Print a JSON artifact when the script is executed directly.
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(json.dumps(run_benchmark(), indent=2, ensure_ascii=False))
