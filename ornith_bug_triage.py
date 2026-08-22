"""Exercise 3: classify bug reports with structured local-model output.

Exercise summary
----------------
Development teams often spend time reading incoming bug reports, assigning a
component and severity, and writing a first diagnostic action. This example
uses the locally hosted Ornith model to perform that repetitive first pass.

The exercise is designed as a measurable benchmark rather than a subjective
demo. Four bug reports have known component and severity labels. Ornith must
return JSON that follows a schema, and Python checks both schema validity and
label accuracy. The final payload also records latency and token usage.

Why this can save time and API cost
-----------------------------------
A local model can triage a large queue before a developer reviews it. Valid,
high-confidence results can be accepted or sampled, while uncertain cases can
be escalated. Local inference has hardware and electricity costs, but it has no
per-request hosted API charge and keeps internal bug text on the machine.

Steps in this example
---------------------
1. Import the required libraries.
2. Configure the LM Studio endpoint, model, and expected output schema.
3. Define labeled bug reports for objective evaluation.
4. Create the OpenAI-compatible LM Studio client.
5. Build a focused prompt for one bug report.
6. Request schema-constrained JSON from Ornith.
7. Validate and score the returned classification.
8. Repeat the process for every benchmark case.
9. Summarize quality, latency, and token usage.
10. Print a JSON artifact when the script is run directly.

Run with ``python ornith_bug_triage.py`` while LM Studio is serving Ornith at
``http://127.0.0.1:1234``. Redirect stdout to a file to preserve the payload.
"""

# Step 1: Import the required libraries.
from __future__ import annotations

import json
import os
import sys
from time import perf_counter
from typing import Any

from openai import OpenAI, OpenAIError


# Step 2: Configure LM Studio and describe the required JSON response.
BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.getenv("LM_STUDIO_MODEL", "ornith-1.0-9b")
API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

# LM Studio can constrain generation with a JSON schema. The schema makes the
# response easy for software to parse and prevents labels outside our approved
# component and severity lists.
TRIAGE_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "bug_triage_result",
        "schema": {
            "type": "object",
            "properties": {
                "report_id": {"type": "string"},
                "component": {
                    "type": "string",
                    "enum": [
                        "authentication",
                        "payments",
                        "frontend",
                        "data",
                        "infrastructure",
                    ],
                },
                "severity": {
                    "type": "string",
                    "enum": ["low", "medium", "high", "critical"],
                },
                "summary": {"type": "string"},
                "next_action": {"type": "string"},
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": [
                "report_id",
                "component",
                "severity",
                "summary",
                "next_action",
                "confidence",
            ],
            "additionalProperties": False,
        },
    },
}


# Step 3: Define labeled reports so model quality can be scored automatically.
#
# These are synthetic examples and contain no private production information.
BUG_REPORTS = [
    {
        "report_id": "AUTH-101",
        "text": (
            "After an access token expires, every tenant receives HTTP 500 "
            "from the refresh endpoint and users cannot sign in again."
        ),
        "expected_component": "authentication",
        "expected_severity": "high",
    },
    {
        "report_id": "PAY-201",
        "text": (
            "A customer can be charged twice by clicking Retry after checkout "
            "times out, even though the first payment succeeded."
        ),
        "expected_component": "payments",
        "expected_severity": "high",
    },
    {
        "report_id": "UI-301",
        "text": "The settings tooltip misspells the word synchronization.",
        "expected_component": "frontend",
        "expected_severity": "low",
    },
    {
        "report_id": "DATA-401",
        "text": (
            "The nightly export omits records created between 00:00 and "
            "00:05 UTC, but the records remain available in the application."
        ),
        "expected_component": "data",
        "expected_severity": "medium",
    },
]


# Step 4: Create an OpenAI-compatible client that sends requests to LM Studio.
def create_client() -> OpenAI:
    """Return a client configured for the local LM Studio endpoint."""

    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


# Step 5: Build one prompt from a benchmark report.
def build_prompt(report: dict[str, str]) -> str:
    """Return the instructions and bug text supplied to the model."""

    return (
        "Triage the synthetic software bug report below. Choose the component "
        "and severity from the allowed schema values. Summarize the problem, "
        "recommend one concrete diagnostic action, and report confidence from "
        "0 to 1. Do not invent facts that are not in the report.\n\n"
        f"Report ID: {report['report_id']}\n"
        f"Bug report: {report['text']}"
    )


# Step 6: Request and parse schema-constrained JSON for one report.
def triage_report(client: OpenAI, report: dict[str, str]) -> dict[str, Any]:
    """Call Ornith once and return its parsed result plus request metrics."""

    started = perf_counter()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a careful software bug-triage assistant.",
            },
            {"role": "user", "content": build_prompt(report)},
        ],
        response_format=TRIAGE_RESPONSE_FORMAT,
        temperature=0.1,
    )
    elapsed_seconds = perf_counter() - started

    content = completion.choices[0].message.content or "{}"
    result = json.loads(content)
    usage = completion.usage.model_dump() if completion.usage else None

    return {
        "result": result,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "usage": usage,
    }


# Step 7: Validate the parsed object and compare it with the known labels.
def evaluate_result(
    report: dict[str, str], result: dict[str, Any]
) -> dict[str, bool]:
    """Return schema and label checks for one model result."""

    required_fields = {
        "report_id",
        "component",
        "severity",
        "summary",
        "next_action",
        "confidence",
    }
    schema_valid = (
        required_fields == set(result)
        and result.get("report_id") == report["report_id"]
        and isinstance(result.get("summary"), str)
        and isinstance(result.get("next_action"), str)
        and isinstance(result.get("confidence"), (int, float))
        and 0 <= result.get("confidence", -1) <= 1
    )

    return {
        "schema_valid": schema_valid,
        "component_correct": (
            result.get("component") == report["expected_component"]
        ),
        "severity_correct": result.get("severity") == report["expected_severity"],
    }


# Step 8: Run every benchmark case while preserving individual failures.
def run_benchmark() -> dict[str, Any]:
    """Triage all reports and return detailed and aggregate results."""

    client = create_client()
    cases: list[dict[str, Any]] = []

    for report in BUG_REPORTS:
        try:
            model_run = triage_report(client, report)
            evaluation = evaluate_result(report, model_run["result"])
            cases.append(
                {
                    "report": report,
                    "model_result": model_run["result"],
                    "evaluation": evaluation,
                    "elapsed_seconds": model_run["elapsed_seconds"],
                    "usage": model_run["usage"],
                    "error": None,
                }
            )
        except (OpenAIError, json.JSONDecodeError, KeyError, TypeError) as error:
            # Continuing after one failure lets a benchmark reveal a partial
            # success rate instead of losing every earlier result.
            cases.append(
                {
                    "report": report,
                    "model_result": None,
                    "evaluation": {
                        "schema_valid": False,
                        "component_correct": False,
                        "severity_correct": False,
                    },
                    "elapsed_seconds": None,
                    "usage": None,
                    "error": str(error),
                }
            )

    # Step 9: Aggregate metrics across all completed cases.
    total_cases = len(cases)
    completed_cases = sum(case["error"] is None for case in cases)
    total_seconds = sum(case["elapsed_seconds"] or 0 for case in cases)
    total_tokens = sum(
        (case["usage"] or {}).get("total_tokens", 0) for case in cases
    )

    return {
        "exercise": "structured_bug_report_triage",
        "backend": "local_lm_studio",
        "model": MODEL,
        "base_url": BASE_URL,
        "cases": cases,
        "summary": {
            "total_cases": total_cases,
            "completed_cases": completed_cases,
            "schema_valid_cases": sum(
                case["evaluation"]["schema_valid"] for case in cases
            ),
            "component_correct_cases": sum(
                case["evaluation"]["component_correct"] for case in cases
            ),
            "severity_correct_cases": sum(
                case["evaluation"]["severity_correct"] for case in cases
            ),
            "elapsed_seconds": round(total_seconds, 3),
            "total_tokens": total_tokens,
            "marginal_hosted_api_cost_usd": 0,
            "local_cost_note": (
                "The API charge is zero; hardware, electricity, and developer "
                "time are outside this simple benchmark."
            ),
        },
    }


# Step 10: Print the benchmark payload as JSON when run directly.
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(json.dumps(run_benchmark(), indent=2, ensure_ascii=False))
