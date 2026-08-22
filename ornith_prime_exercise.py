"""Submit one prompt to the Ornith model running in LM Studio.

The script returns the model response as a JSON-serializable payload. It does
not start an interactive command-line chat.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI


BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.getenv("LM_STUDIO_MODEL", "ornith-1.0-9b")
PROMPT = (
    "Please write an annotated R for data science custom function that returns "
    "the nearest prime below and above a given number, using tidyverse syntax"
)


def run_exercise() -> dict[str, Any]:
    """Send the exercise prompt to LM Studio and return the result payload."""

    client = OpenAI(
        api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        base_url=BASE_URL,
    )

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.2,
    )

    choice = completion.choices[0]
    usage = completion.usage

    return {
        "request": {
            "base_url": BASE_URL,
            "model": MODEL,
            "prompt": PROMPT,
        },
        "response": {
            "id": completion.id,
            "model": completion.model,
            "content": choice.message.content,
            "finish_reason": choice.finish_reason,
            "usage": usage.model_dump() if usage is not None else None,
        },
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(run_exercise(), indent=2, ensure_ascii=False))
