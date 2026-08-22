"""Exercise: send one teaching prompt to a local language model.

Exercise summary
----------------
This example teaches the basic flow for calling a language model hosted by
LM Studio from Python. The script submits one prompt to the Ornith 1.0 9B
model, receives the model's answer, and returns both the request details and
the response in a JSON-serializable Python dictionary.

The example is intentionally a single submission rather than an interactive
command-line chat. That makes the input, output, and API call easy to inspect
and reuse in a larger program.

Steps in this example
---------------------
1. Import the Python libraries used by the script.
2. Define the LM Studio connection settings and exercise prompt.
3. Define a function that owns the complete model-call workflow.
4. Create an OpenAI-compatible client pointed at LM Studio.
5. Submit the prompt to the model through the chat-completions API.
6. Extract the useful parts of the API response.
7. Assemble and return a structured payload.
8. When run as a script, print the payload as readable JSON.
"""

# Step 1: Import the libraries used by the example.
#
# `json` converts the final Python dictionary into readable JSON text.
# `os` lets users override the defaults with environment variables.
# `sys` lets the script request UTF-8 output on terminals that support it.
# `Any` is used in the return type because the API usage object contains
# several numeric fields whose exact shape is defined by the OpenAI client.
from __future__ import annotations

import json
import os
import sys
from typing import Any

# LM Studio exposes an API that follows the OpenAI client interface. Installing
# the `openai` package therefore gives us a ready-made Python client instead of
# requiring us to build HTTP requests by hand.
from openai import OpenAI


# Step 2: Define the connection settings and the exercise prompt.
#
# LM Studio normally listens on port 1234. The `/v1` suffix selects its
# OpenAI-compatible API routes. Environment variables are optional, but they
# let the same script work with a different server address or model name:
#
#   $env:LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
#   $env:LM_STUDIO_MODEL = "ornith-1.0-9b"
#
# The fallback values make the script work with the local setup used in this
# exercise without requiring any configuration first.
BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.getenv("LM_STUDIO_MODEL", "ornith-1.0-9b")

# Keeping the prompt in a named constant makes it easy to read, change, or
# reuse. It also lets us include the exact submitted prompt in the return
# payload so that the output remains self-describing.
PROMPT = (
    "Please write an annotated R for data science custom function that returns "
    "the nearest prime below and above a given number, using tidyverse syntax"
)


# Step 3: Define the complete model-call workflow as a reusable function.
#
# The function returns a dictionary instead of printing directly. Returning
# data makes this code useful from another Python module, a test, or a web
# application. The `dict[str, Any]` annotation tells readers that the result
# is a dictionary whose values may have different valid types, such as strings,
# numbers, dictionaries, or None.
def run_exercise() -> dict[str, Any]:
    """Submit the exercise prompt and return a self-describing payload."""

    # Step 4: Create an OpenAI-compatible client connected to LM Studio.
    #
    # LM Studio does not need a real cloud API key for a local request, but the
    # OpenAI client requires a non-empty value. The placeholder is appropriate
    # for this local LM Studio setup. If BASE_URL is changed to a hosted
    # service, set LM_STUDIO_API_KEY to the real key required by that service.
    client = OpenAI(
        api_key=os.getenv("LM_STUDIO_API_KEY", "lm-studio"),
        base_url=BASE_URL,
    )

    # Step 5: Submit one chat completion request.
    #
    # `messages` is a list because chat models support conversations with
    # multiple roles. This first exercise has one user message. The low
    # temperature asks for a relatively consistent answer while still using
    # the model's normal text-generation behavior.
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": PROMPT}],
        temperature=0.2,
    )

    # Step 6: Extract the response objects we want to preserve.
    #
    # A completion may contain multiple choices. This exercise requests the
    # first choice, which is the usual pattern for a single-answer prompt.
    # Usage can be absent for some compatible servers, so it is retained only
    # when the server provides it.
    choice = completion.choices[0]
    usage = completion.usage

    # Step 7: Build and return a structured payload.
    #
    # Keeping request metadata beside the model response makes the artifact
    # useful later: a reader can see which model was called, where it was
    # called, what prompt was sent, and what answer came back.
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


# Step 8: Run the exercise when this file is executed directly.
#
# The `__name__` check keeps this block from running when another Python file
# imports `run_exercise()`. In that case, the caller receives the dictionary
# and can decide how to store, inspect, or display it.
if __name__ == "__main__":
    # Some Windows terminals use a legacy encoding that cannot display every
    # character a model may return. Reconfiguring stdout prevents a successful
    # API response from failing only at the final printing step.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Convert the returned dictionary to indented JSON so a person can read it
    # in the terminal or redirect it into an artifact file. `ensure_ascii=False`
    # preserves the model's original Unicode characters when possible.
    print(json.dumps(run_exercise(), indent=2, ensure_ascii=False))
