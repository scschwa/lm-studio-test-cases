"""Exercise: build a basic interactive chat with a local LM Studio model.

Exercise summary
----------------
This example teaches how to create a terminal application that holds a
back-and-forth conversation with the Ornith 1.0 9B model running in LM Studio.
The user enters one message at a time, the script sends the conversation to
the model, and the model's reply is printed in the terminal.

The central idea is conversation history. A language-model API does not
automatically remember earlier requests made by this Python process. To give
the model context, the script keeps every user and assistant message in a
Python list and sends that list with each new request.

This is intentionally a low-frills teaching example. It uses non-streaming
responses, keeps its history only in memory, and ends when the user enters
``exit`` or ``quit``. A production application would usually add history
limits, persistence, logging, streaming, and more detailed error handling.

Steps in this example
---------------------
1. Import the required Python libraries.
2. Define the LM Studio connection and chat settings.
3. Create an OpenAI-compatible client pointed at LM Studio.
4. Define a helper that submits the current conversation.
5. Start the conversation history with a system message.
6. Read and validate each terminal message from the user.
7. Add the user's message to the conversation history.
8. Send the full history to the model and handle request errors.
9. Add and display the assistant's reply.
10. Configure terminal output and start the chat when run directly.

Run the exercise
----------------
1. Start the local server in LM Studio and load ``ornith-1.0-9b``.
2. Install the dependency with ``python -m pip install -r requirements.txt``.
3. Run this file with ``python ornith_terminal_chat.py``.
4. Enter ``exit`` or ``quit`` when you are finished.
"""

# Step 1: Import the libraries used by the chat application.
from __future__ import annotations

import os
import sys

# LM Studio implements OpenAI-compatible endpoints, so the OpenAI Python
# package can communicate with the local server. APIConnectionError represents
# network-level failures, while APIError covers other errors returned during an
# API request.
from openai import APIConnectionError, APIError, OpenAI


# Step 2: Define the connection and chat settings.
#
# The default base URL points to LM Studio's OpenAI-compatible API. Environment
# variables allow someone to change the address, model, or API key without
# editing the teaching example itself.
BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.getenv("LM_STUDIO_MODEL", "ornith-1.0-9b")
API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")

# A system message sets the model's overall behavior for the conversation. It
# becomes the first entry in the history and is sent again with every request.
SYSTEM_PROMPT = "You are a helpful assistant. Explain technical ideas clearly."

# These commands are checked locally. They are not sent to the model.
EXIT_COMMANDS = {"exit", "quit"}


# Step 3: Create an OpenAI-compatible client pointed at LM Studio.
def create_client() -> OpenAI:
    """Return a client configured to call the local LM Studio server."""

    # The OpenAI client requires an API-key value. LM Studio does not require a
    # real cloud key for this local setup, so "lm-studio" is a safe placeholder.
    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


# Step 4: Define a helper that submits the current conversation.
def request_reply(client: OpenAI, messages: list[dict[str, str]]) -> str:
    """Send the conversation history and return the assistant's text reply."""

    # `messages` contains the system instruction plus every prior user and
    # assistant turn. Sending the full list gives the otherwise stateless API
    # the context it needs to continue the conversation.
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.7,
    )

    # This example requests one answer, so the first choice is the reply we
    # want. Some API responses permit content to be None, so an empty string is
    # used as a safe fallback for display and history storage.
    return completion.choices[0].message.content or ""


def run_chat() -> None:
    """Run the terminal input, model request, and response loop."""

    client = create_client()

    # Step 5: Start conversation history with the system message.
    #
    # This list exists only while the program is running. Closing the program
    # discards the history. Each following loop iteration appends one user turn
    # and one assistant turn.
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print(f"Connected to {MODEL} through {BASE_URL}")
    print("Type a message and press Enter. Type 'exit' or 'quit' to finish.")

    while True:
        # Step 6: Read and validate one terminal message from the user.
        #
        # Ctrl+C raises KeyboardInterrupt and Ctrl+Z followed by Enter on
        # Windows raises EOFError. Treating both as normal exits avoids showing
        # an unnecessary traceback when the user wants to stop.
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding chat.")
            return

        # Ignore an empty submission because it adds no useful chat content.
        if not user_text:
            continue

        # Exit commands are handled before appending the message, which keeps
        # them out of the model's conversation history.
        if user_text.lower() in EXIT_COMMANDS:
            print("Ending chat.")
            return

        # Step 7: Add the user's message to the conversation history.
        messages.append({"role": "user", "content": user_text})

        # Step 8: Send the full conversation and handle request failures.
        try:
            assistant_text = request_reply(client, messages)
        except APIConnectionError:
            # Remove the unsent user turn so retrying does not accidentally add
            # it to the conversation twice.
            messages.pop()
            print(
                "\nCould not connect to LM Studio. Confirm that its local "
                f"server is running at {BASE_URL}."
            )
            continue
        except APIError as error:
            messages.pop()
            print(f"\nLM Studio returned an API error: {error}")
            continue
        except KeyboardInterrupt:
            messages.pop()
            print("\nRequest interrupted. Ending chat.")
            return

        # Step 9: Store and display the assistant's reply.
        #
        # Saving the reply is what lets a later user message refer to it. The
        # next request will include this assistant turn along with the rest of
        # the history.
        messages.append({"role": "assistant", "content": assistant_text})
        print(f"\nOrnith: {assistant_text}")

        # This simple example never removes old messages. Long conversations
        # will eventually approach the model's context limit and take longer
        # to process because every request resends the accumulated history.


# Step 10: Configure output and start the chat when this file runs directly.
if __name__ == "__main__":
    # UTF-8 output prevents model responses containing characters outside the
    # Windows legacy console encoding from failing during display.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Keeping execution behind this check allows another Python module to
    # import create_client(), request_reply(), or run_chat() without
    # immediately starting an interactive prompt.
    run_chat()
