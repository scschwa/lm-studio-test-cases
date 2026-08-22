# LM Studio Local Model Test Cases

This repository contains teaching-oriented Python examples for using a language
model hosted locally by [LM Studio](https://lmstudio.ai/). The examples use the
OpenAI-compatible Python client, so the same request patterns can be adapted to
other compatible local or hosted services by changing configuration values.

The examples progress from a single model request to interactive chat,
structured evaluation, tool calling, local-first routing, and collaborative
synthetic-data generation. Each script includes a numbered walkthrough in the
source code. Representative execution results are preserved in `artifacts/`.

## Local model configuration

The committed examples use these defaults:

- LM Studio server: `http://127.0.0.1:1234/v1`
- Model identifier: `ornith-1.0-9b`
- Python client: `openai`

Before running an example:

1. Open LM Studio and load the Ornith 1.0 9B model.
2. Start LM Studio's local API server.
3. Open PowerShell in this repository.
4. Install the Python dependency:

```powershell
python -m pip install -r requirements.txt
```

The defaults can be overridden without editing the scripts:

```powershell
$env:LM_STUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
$env:LM_STUDIO_MODEL = "ornith-1.0-9b"
$env:LM_STUDIO_API_KEY = "lm-studio"
```

## Example 1: Single prompt and structured return payload

`ornith_prime_exercise.py` sends one prompt asking Ornith to create an annotated
R function that finds the nearest prime below and above a number. It returns
request metadata, model output, finish reason, and token usage as JSON.

```powershell
python ornith_prime_exercise.py
```

To preserve another run:

```powershell
python ornith_prime_exercise.py > artifacts/ornith_prime_exercise_output.json
```

## Example 2: Interactive terminal chat

`ornith_terminal_chat.py` maintains a conversation by appending each user and
assistant message to an in-memory history list. Enter `exit` or `quit` to stop.

```powershell
python ornith_terminal_chat.py
```

## Example 3: Structured bug-report triage

`ornith_bug_triage.py` classifies four labeled synthetic bug reports. It uses a
JSON schema and measures response validity, component accuracy, severity
accuracy, latency, and token usage.

```powershell
python ornith_bug_triage.py
```

To save a new benchmark artifact:

```powershell
python ornith_bug_triage.py > artifacts/ornith_bug_triage_output.json
```

## Example 4: Unit-test generation and bug detection

`ornith_unit_test_generation.py` asks Ornith to generate structured test inputs
and expected values for two deliberately defective functions. Trusted Python
code executes the test data and measures whether the generated cases expose the
known bugs. The application does not execute arbitrary source code returned by
the model.

```powershell
python ornith_unit_test_generation.py
```

To save a new benchmark artifact:

```powershell
python ornith_unit_test_generation.py > artifacts/ornith_unit_test_generation_output.json
```

## Example 5: Tool-calling accuracy

`ornith_tool_calling_benchmark.py` gives the model three safe local tools and
tests tool selection, argument construction, multiple tool requests, and a case
where no tool should be used. Python executes only functions in an explicit
allowlist.

```powershell
python ornith_tool_calling_benchmark.py
```

To save a new benchmark artifact:

```powershell
python ornith_tool_calling_benchmark.py > artifacts/ornith_tool_calling_benchmark_output.json
```

## Example 6: Local-first hosted escalation

`ornith_local_first_escalation.py` sends every request to Ornith first. A
deterministic policy accepts valid routine results locally and escalates
security, financial, architectural, invalid, or low-confidence results to a
hosted model. It also runs an all-hosted baseline to estimate avoided requests
and token cost.

This example makes paid hosted API requests. Configure the hosted key and, if
needed, override the model and current pricing assumptions:

```powershell
$env:OPENAI_API_KEY = "your-hosted-api-key"
$env:HOSTED_MODEL = "gpt-5.4-nano"
$env:HOSTED_INPUT_USD_PER_MILLION = "0.20"
$env:HOSTED_OUTPUT_USD_PER_MILLION = "1.25"
python ornith_local_first_escalation.py
```

To save a new benchmark artifact:

```powershell
python ornith_local_first_escalation.py > artifacts/ornith_local_first_escalation_output.json
```

Verify hosted pricing before using the cost comparison. The script does not
write the API key into its output.

## Example 7: Interactive fake dataset creator

`ornith_fake_dataset_creator.py` interviews the user about a fictional dataset,
asks Ornith to propose a field plan, supports approval or revision, generates
schema-constrained records, validates them in Python, and writes one or more
CSV, JSON, or XML files.

```powershell
python ornith_fake_dataset_creator.py
```

By default, generated files and a run manifest are written beneath
`artifacts/generated_datasets/`. To choose another output root:

```powershell
$env:DATASET_OUTPUT_ROOT = "generated_datasets"
python ornith_fake_dataset_creator.py
```

The application limits a run to 100 records, 20 files, and scalar field types.
Generated synthetic data should still be reviewed before consequential use.

## Preserved artifacts

The `artifacts/` directory contains representative outputs from completed runs:

- One-shot model response
- Bug-triage benchmark results
- Unit-test generation benchmark results
- Tool-calling benchmark results
- Local-first escalation comparison
- Fake-dataset CLI transcript, manifest, and generated CSV files

Outputs are model-generated and can vary between runs, even when prompts and
settings remain unchanged.
