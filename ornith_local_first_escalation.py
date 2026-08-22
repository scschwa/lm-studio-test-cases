"""Exercise 6: route routine work locally and escalate difficult cases.

Exercise summary
----------------
This example demonstrates a local-first model cascade. Every synthetic
development request goes to Ornith first. Python then applies a transparent
routing policy: routine, valid, high-confidence results are accepted locally,
while security, financial, architectural, invalid, or low-confidence results
are escalated to a hosted model.

For benchmarking, the script also obtains a hosted answer for every case. That
all-hosted baseline lets us use actual hosted token counts to compare two
strategies:

* All-hosted: send every request directly to the hosted API.
* Local-first: use hosted results only for cases the router escalates.

The benchmark itself incurs the all-hosted baseline cost because it calls the
hosted model for measurement. The reported local-first cost is a projection
calculated from the subset of those calls that routing would have required.

Why this can save time and API cost
-----------------------------------
A local model does not need to replace a hosted model on every task. It can
handle repetitive, low-risk work and reserve paid calls for difficult cases.
This pattern can reduce hosted usage while preserving a clear path to stronger
reasoning when risk or uncertainty is high.

Steps in this example
---------------------
1. Import the required libraries.
2. Configure local and hosted clients plus pricing assumptions.
3. Define a shared structured-output schema.
4. Define synthetic requests with known categories.
5. Request an assessment from either backend.
6. Validate each model response.
7. Apply the local-first routing policy.
8. Run local and hosted baseline requests.
9. Calculate quality, usage, and projected savings.
10. Print a JSON artifact without exposing API credentials.

Required configuration
----------------------
LM Studio must be serving Ornith locally. ``OPENAI_API_KEY`` must be set for
the hosted baseline. ``HOSTED_MODEL`` and the two hosted price environment
variables can override the defaults when models or prices change.
"""

# Step 1: Import libraries for model clients, timing, JSON, and configuration.
from __future__ import annotations

import json
import os
import sys
from time import perf_counter
from typing import Any

from openai import OpenAI, OpenAIError


# Step 2: Configure local and hosted endpoints and pricing assumptions.
LOCAL_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
LOCAL_MODEL = os.getenv("LM_STUDIO_MODEL", "ornith-1.0-9b")
LOCAL_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

# The hosted client uses the OpenAI package's default hosted endpoint. Keeping
# the model configurable makes the exercise reusable as product names change.
HOSTED_MODEL = os.getenv("HOSTED_MODEL", "gpt-5.4-nano")
HOSTED_API_KEY = os.getenv("OPENAI_API_KEY")

# Rates are dollars per one million tokens. They are benchmark assumptions,
# not universal constants. Override them from the environment after checking
# the provider's current pricing page.
HOSTED_INPUT_USD_PER_MILLION = float(
    os.getenv("HOSTED_INPUT_USD_PER_MILLION", "0.20")
)
HOSTED_OUTPUT_USD_PER_MILLION = float(
    os.getenv("HOSTED_OUTPUT_USD_PER_MILLION", "1.25")
)


# Step 3: Define one schema so local and hosted responses have the same shape.
ASSESSMENT_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "development_request_assessment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string"},
                "category": {
                    "type": "string",
                    "enum": ["routine", "security", "financial", "architecture"],
                },
                "recommendation": {"type": "string"},
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "requires_expert_review": {"type": "boolean"},
            },
            "required": [
                "request_id",
                "category",
                "recommendation",
                "confidence",
                "requires_expert_review",
            ],
            "additionalProperties": False,
        },
    },
}


# Step 4: Define synthetic requests with known expected categories.
REQUESTS = [
    {
        "request_id": "DEV-101",
        "text": (
            "Correct one spelling mistake in static tooltip text on the account "
            "settings page. No logic or design changes are involved."
        ),
        "expected_category": "routine",
    },
    {
        "request_id": "DEV-102",
        "text": "Add docstrings to five small internal string helper functions.",
        "expected_category": "routine",
    },
    {
        "request_id": "SEC-201",
        "text": (
            "Review an authentication flow where changing an account ID in the "
            "URL may expose another customer's profile."
        ),
        "expected_category": "security",
    },
    {
        "request_id": "FIN-301",
        "text": (
            "Recommend a fix for checkout retries that can create duplicate "
            "customer charges after a network timeout."
        ),
        "expected_category": "financial",
    },
    {
        "request_id": "ARCH-401",
        "text": (
            "Choose a data consistency strategy for a new multi-region order "
            "system with regional failover."
        ),
        "expected_category": "architecture",
    },
]

# Categories with material risk or broad design consequences are always sent
# to the hosted model, even if the local model reports high confidence.
ALWAYS_ESCALATE_CATEGORIES = {"security", "financial", "architecture"}
LOCAL_CONFIDENCE_THRESHOLD = 0.80


def create_local_client() -> OpenAI:
    """Return the OpenAI-compatible LM Studio client."""

    return OpenAI(base_url=LOCAL_BASE_URL, api_key=LOCAL_API_KEY)


def create_hosted_client() -> OpenAI | None:
    """Return a hosted client when a key is configured, otherwise None."""

    if not HOSTED_API_KEY:
        return None
    return OpenAI(api_key=HOSTED_API_KEY)


# Step 5: Request the same structured assessment from either backend.
def request_assessment(
    client: OpenAI,
    model: str,
    request: dict[str, str],
) -> dict[str, Any]:
    """Return a parsed assessment with latency and usage metadata."""

    prompt = (
        "Assess this synthetic software-development request. Classify its "
        "category, give one concise recommendation, estimate confidence, and "
        "state whether expert review is appropriate. Preserve the request ID. "
        "Use routine for small mechanical changes. Confidence MUST be a decimal "
        "from 0.0 through 1.0, never a percentage from 0 through 100.\n\n"
        f"Request ID: {request['request_id']}\n"
        f"Request: {request['text']}"
    )

    started = perf_counter()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a careful software-development request reviewer.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=ASSESSMENT_RESPONSE_FORMAT,
    )
    elapsed_seconds = perf_counter() - started

    content = completion.choices[0].message.content or "{}"
    return {
        "assessment": json.loads(content),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "usage": completion.usage.model_dump() if completion.usage else None,
    }


# Step 6: Normalize a common small-model variation, then validate routing fields.
def normalize_assessment(
    assessment: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Convert a percentage-like confidence to the documented 0-to-1 scale."""

    normalized = dict(assessment)
    notes: list[str] = []
    confidence = normalized.get("confidence")

    # Some small models return 85 when a schema requests 0.85. Accepting this
    # narrowly defined variation makes the router resilient while preserving a
    # note in the artifact. Values outside 0 through 100 are never repaired.
    if isinstance(confidence, (int, float)) and 1 < confidence <= 100:
        normalized["confidence"] = confidence / 100
        notes.append(f"normalized confidence from {confidence} to {confidence / 100}")

    return normalized, notes


def validate_assessment(
    request: dict[str, str], assessment: dict[str, Any]
) -> bool:
    """Return True when the parsed response is safe for routing decisions."""

    required_fields = {
        "request_id",
        "category",
        "recommendation",
        "confidence",
        "requires_expert_review",
    }
    return (
        set(assessment) == required_fields
        and assessment.get("request_id") == request["request_id"]
        and assessment.get("category")
        in {"routine", "security", "financial", "architecture"}
        and isinstance(assessment.get("recommendation"), str)
        and isinstance(assessment.get("confidence"), (int, float))
        and 0 <= assessment.get("confidence", -1) <= 1
        and isinstance(assessment.get("requires_expert_review"), bool)
    )


# Step 7: Apply an explicit policy instead of trusting model confidence alone.
def route_local_result(
    schema_valid: bool, assessment: dict[str, Any] | None
) -> tuple[str, list[str]]:
    """Return the selected backend and human-readable routing reasons."""

    reasons: list[str] = []
    if not schema_valid or assessment is None:
        reasons.append("local response was missing or invalid")
    else:
        if assessment["category"] in ALWAYS_ESCALATE_CATEGORIES:
            reasons.append(f"{assessment['category']} is an always-escalate category")
        if assessment["confidence"] < LOCAL_CONFIDENCE_THRESHOLD:
            reasons.append(
                f"confidence {assessment['confidence']} is below "
                f"{LOCAL_CONFIDENCE_THRESHOLD}"
            )
        if assessment["requires_expert_review"]:
            reasons.append("local model requested expert review")

    if reasons:
        return "hosted", reasons
    return "local", ["valid routine result met the local acceptance policy"]


def hosted_cost(usage: dict[str, Any] | None) -> float:
    """Calculate hosted token cost using the configured benchmark rates."""

    if usage is None:
        return 0.0
    input_cost = usage.get("prompt_tokens", 0) * HOSTED_INPUT_USD_PER_MILLION
    output_cost = usage.get("completion_tokens", 0) * HOSTED_OUTPUT_USD_PER_MILLION
    return (input_cost + output_cost) / 1_000_000


# Step 8: Run every case locally and obtain an all-hosted comparison baseline.
def run_benchmark() -> dict[str, Any]:
    """Return routed results and a measured all-hosted cost comparison."""

    local_client = create_local_client()
    hosted_client = create_hosted_client()
    case_results: list[dict[str, Any]] = []

    for request in REQUESTS:
        local_result: dict[str, Any] | None = None
        local_error: str | None = None
        try:
            local_result = request_assessment(local_client, LOCAL_MODEL, request)
        except (OpenAIError, json.JSONDecodeError, KeyError, TypeError) as error:
            local_error = str(error)

        local_raw_assessment = (
            local_result["assessment"] if local_result is not None else None
        )
        if local_raw_assessment is not None:
            local_assessment, local_normalizations = normalize_assessment(
                local_raw_assessment
            )
        else:
            local_assessment, local_normalizations = None, []
        local_valid = (
            validate_assessment(request, local_assessment)
            if local_assessment is not None
            else False
        )
        selected_backend, routing_reasons = route_local_result(
            local_valid, local_assessment
        )

        hosted_result: dict[str, Any] | None = None
        hosted_error: str | None = None
        if hosted_client is not None:
            try:
                hosted_result = request_assessment(
                    hosted_client, HOSTED_MODEL, request
                )
            except (OpenAIError, json.JSONDecodeError, KeyError, TypeError) as error:
                hosted_error = str(error)
        else:
            hosted_error = "OPENAI_API_KEY is not configured"

        hosted_raw_assessment = (
            hosted_result["assessment"] if hosted_result is not None else None
        )
        if hosted_raw_assessment is not None:
            hosted_assessment, hosted_normalizations = normalize_assessment(
                hosted_raw_assessment
            )
        else:
            hosted_assessment, hosted_normalizations = None, []
        hosted_valid = (
            validate_assessment(request, hosted_assessment)
            if hosted_assessment is not None
            else False
        )
        hosted_request_cost = hosted_cost(
            hosted_result["usage"] if hosted_result is not None else None
        )

        case_results.append(
            {
                "request": request,
                "local": {
                    **{
                        key: value
                        for key, value in (local_result or {}).items()
                        if key != "assessment"
                    },
                    "raw_assessment": local_raw_assessment,
                    "assessment": local_assessment,
                    "normalizations": local_normalizations,
                    "schema_valid": local_valid,
                    "category_correct": (
                        local_assessment is not None
                        and local_assessment.get("category")
                        == request["expected_category"]
                    ),
                    "error": local_error,
                },
                "routing": {
                    "selected_backend": selected_backend,
                    "reasons": routing_reasons,
                },
                "hosted_baseline": {
                    **{
                        key: value
                        for key, value in (hosted_result or {}).items()
                        if key != "assessment"
                    },
                    "raw_assessment": hosted_raw_assessment,
                    "assessment": hosted_assessment,
                    "normalizations": hosted_normalizations,
                    "schema_valid": hosted_valid,
                    "category_correct": (
                        hosted_assessment is not None
                        and hosted_assessment.get("category")
                        == request["expected_category"]
                    ),
                    "estimated_cost_usd": round(hosted_request_cost, 8),
                    "error": hosted_error,
                },
            }
        )

    # Step 9: Compare measured all-hosted usage with projected routed usage.
    all_hosted_cost = sum(
        case["hosted_baseline"]["estimated_cost_usd"] for case in case_results
    )
    local_first_hosted_cost = sum(
        case["hosted_baseline"]["estimated_cost_usd"]
        for case in case_results
        if case["routing"]["selected_backend"] == "hosted"
    )

    return {
        "exercise": "local_first_escalation",
        "local_backend": {
            "base_url": LOCAL_BASE_URL,
            "model": LOCAL_MODEL,
        },
        "hosted_backend": {
            "model": HOSTED_MODEL,
            "api_key_configured": HOSTED_API_KEY is not None,
            "pricing_assumption_usd_per_million_tokens": {
                "input": HOSTED_INPUT_USD_PER_MILLION,
                "output": HOSTED_OUTPUT_USD_PER_MILLION,
            },
        },
        "benchmark_note": (
            "The run called the hosted model for every case to measure an "
            "all-hosted baseline. Projected local-first cost counts only cases "
            "that the routing policy selected for escalation."
        ),
        "cases": case_results,
        "summary": {
            "total_cases": len(case_results),
            "accepted_locally": sum(
                case["routing"]["selected_backend"] == "local"
                for case in case_results
            ),
            "escalated_to_hosted": sum(
                case["routing"]["selected_backend"] == "hosted"
                for case in case_results
            ),
            "local_schema_valid_cases": sum(
                case["local"]["schema_valid"] for case in case_results
            ),
            "local_category_correct_cases": sum(
                case["local"]["category_correct"] for case in case_results
            ),
            "hosted_completed_cases": sum(
                case["hosted_baseline"]["error"] is None for case in case_results
            ),
            "hosted_category_correct_cases": sum(
                case["hosted_baseline"]["category_correct"]
                for case in case_results
            ),
            "measured_all_hosted_cost_usd": round(all_hosted_cost, 8),
            "projected_local_first_hosted_cost_usd": round(
                local_first_hosted_cost, 8
            ),
            "projected_hosted_cost_avoided_usd": round(
                all_hosted_cost - local_first_hosted_cost, 8
            ),
            "projected_hosted_requests_avoided": sum(
                case["routing"]["selected_backend"] == "local"
                for case in case_results
            ),
            "local_cost_note": (
                "Local hardware, electricity, and maintenance costs are not "
                "included in the token-price comparison."
            ),
        },
    }


# Step 10: Print results without ever placing the API key in the payload.
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print(json.dumps(run_benchmark(), indent=2, ensure_ascii=False))
