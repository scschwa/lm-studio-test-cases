"""Exercise 7: collaboratively create synthetic datasets from the terminal.

Exercise summary
----------------
This application uses the locally hosted Ornith model as a dataset-design and
generation assistant. The user describes a dataset through a short terminal
interview, Ornith proposes a field plan, and the user can approve that plan or
request revisions. After approval, the model generates structured records and
trusted Python code validates and writes them as CSV, JSON, or XML files.

The application supports one file or a set of files. For example, a user can
request one JSON file, three XML files, or five CSV files. Records are split as
evenly as possible across the requested file count. Every run also writes a
manifest containing the requirements, approved plan, model metadata, generated
file names, token usage, and validation result.

Why use a local model for this task
-----------------------------------
Synthetic-data design is iterative. A developer may revise field descriptions,
generate many small fixtures, and repeat the process throughout testing. Local
inference avoids a hosted charge for each iteration and can keep proprietary
schema descriptions on the workstation. Generated data must still be reviewed
before it is used for consequential testing or analysis.

Safety and scope
----------------
The prompts explicitly request fictional data and discourage real personal
information. This low-frills example supports scalar field types only: string,
integer, number, boolean, and ISO date. It limits each run to 100 records and 20
files so that accidental requests remain manageable for a small local model.

Steps in this example
---------------------
1. Import standard-library and OpenAI-compatible dependencies.
2. Configure LM Studio, output paths, limits, and supported types.
3. Collect dataset requirements through a terminal interview.
4. Define the schema Ornith must use for its proposed dataset plan.
5. Ask Ornith to convert the user's description into a field plan.
6. Normalize names and let the user approve or revise the plan.
7. Build a record schema dynamically from the approved fields.
8. Ask Ornith to generate exactly the requested number of records.
9. Validate record keys, scalar types, and ISO dates in Python.
10. Split records across the requested number of files.
11. Serialize records as CSV, JSON, or XML.
12. Write a manifest and display a concise completion summary.

Run with ``python ornith_fake_dataset_creator.py`` while LM Studio is serving
Ornith. Generated datasets are placed under ``artifacts/generated_datasets``
by default. Set ``DATASET_OUTPUT_ROOT`` to choose another root directory.
"""

# Step 1: Import standard-library tools and the OpenAI-compatible client.
from __future__ import annotations

import csv
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from openai import OpenAI, OpenAIError


# Step 2: Configure LM Studio, output behavior, and intentionally small limits.
BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
MODEL = os.getenv("LM_STUDIO_MODEL", "ornith-1.0-9b")
API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
OUTPUT_ROOT = Path(
    os.getenv("DATASET_OUTPUT_ROOT", "artifacts/generated_datasets")
)

SUPPORTED_FORMATS = {"csv", "json", "xml"}
SUPPORTED_FIELD_TYPES = {"string", "integer", "number", "boolean", "date"}
MAX_RECORDS = 100
MAX_FILES = 20
MAX_FIELDS = 20


class UserCancelled(Exception):
    """Represent a normal user cancellation without a traceback."""


def create_client() -> OpenAI:
    """Return a client connected to the local LM Studio server."""

    return OpenAI(base_url=BASE_URL, api_key=API_KEY)


def read_text(prompt: str, default: str | None = None) -> str:
    """Read non-empty terminal text, applying an optional default."""

    while True:
        default_label = f" [{default}]" if default is not None else ""
        try:
            value = input(f"{prompt}{default_label}: ").strip()
        except (EOFError, KeyboardInterrupt) as error:
            raise UserCancelled from error

        if value:
            return value
        if default is not None:
            return default
        print("Please enter a value.")


def read_integer(prompt: str, default: int, minimum: int, maximum: int) -> int:
    """Read and range-check one integer from the terminal."""

    while True:
        raw_value = read_text(prompt, str(default))
        try:
            value = int(raw_value)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if minimum <= value <= maximum:
            return value
        print(f"Please enter a number from {minimum} through {maximum}.")


def sanitize_name(
    value: str, fallback: str, prefix_leading_digit: bool = False
) -> str:
    """Convert a user or model label into a safe lowercase identifier."""

    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip()).strip("_").lower()
    normalized = re.sub(r"_+", "_", normalized)
    if not normalized:
        normalized = fallback
    # Python-like field names should not begin with a digit. File and directory
    # names may begin with one, so callers opt in to this field-specific rule.
    if prefix_leading_digit and normalized[0].isdigit():
        normalized = f"field_{normalized}"
    return normalized


# Step 3: Collect requirements through a compact terminal interview.
def collect_requirements() -> dict[str, Any]:
    """Ask the user what dataset should be designed and generated."""

    print("Ornith Fake Dataset Creator")
    print("Describe a fictional dataset, review the plan, then generate files.\n")

    dataset_name = sanitize_name(
        read_text("Dataset name", "sample_dataset"), "sample_dataset"
    )
    purpose = read_text("What will this dataset be used to test?")

    while True:
        output_format = read_text("Output format: csv, json, or xml", "csv").lower()
        if output_format in SUPPORTED_FORMATS:
            break
        print("Supported formats are csv, json, and xml.")

    record_count = read_integer("Total number of records", 10, 1, MAX_RECORDS)
    file_count = read_integer(
        "Number of output files",
        1,
        1,
        min(MAX_FILES, record_count),
    )
    field_description = read_text(
        "Describe the fields, types, and important allowed values"
    )
    constraints = read_text(
        "Additional constraints or edge cases", "No additional constraints"
    )

    return {
        "dataset_name": dataset_name,
        "purpose": purpose,
        "format": output_format,
        "record_count": record_count,
        "file_count": file_count,
        "field_description": field_description,
        "constraints": constraints,
    }


# Step 4: Define the structured shape of the model's proposed dataset plan.
PLAN_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "synthetic_dataset_plan",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "dataset_name": {"type": "string"},
                "purpose": {"type": "string"},
                "format": {"type": "string", "enum": ["csv", "json", "xml"]},
                "record_count": {"type": "integer"},
                "file_count": {"type": "integer"},
                "fields": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_FIELDS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": sorted(SUPPORTED_FIELD_TYPES),
                            },
                            "description": {"type": "string"},
                        },
                        "required": ["name", "type", "description"],
                        "additionalProperties": False,
                    },
                },
                "generation_instructions": {"type": "string"},
            },
            "required": [
                "dataset_name",
                "purpose",
                "format",
                "record_count",
                "file_count",
                "fields",
                "generation_instructions",
            ],
            "additionalProperties": False,
        },
    },
}


# Step 5: Ask Ornith to translate natural-language requirements into a plan.
def request_plan(
    client: OpenAI,
    requirements: dict[str, Any],
    revision_notes: list[str],
) -> dict[str, Any]:
    """Return one proposed plan plus timing and usage information."""

    revision_text = (
        "\nRequested revisions:\n- " + "\n- ".join(revision_notes)
        if revision_notes
        else ""
    )
    prompt = (
        "Create a practical plan for a fully fictional synthetic dataset. Use "
        "only scalar fields supported by the schema. Field names should be "
        "short snake_case identifiers. Put allowed categories, uniqueness, date "
        "ranges, and relationships in field descriptions or generation "
        "instructions. Preserve the requested format and counts. Never request "
        "or generate real personal data.\n\n"
        f"User requirements:\n{json.dumps(requirements, indent=2)}"
        f"{revision_text}"
    )

    started = perf_counter()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You design compact, safe, fictional test datasets.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=PLAN_RESPONSE_FORMAT,
        temperature=0.2,
    )
    elapsed_seconds = perf_counter() - started

    content = completion.choices[0].message.content or "{}"
    return {
        "plan": json.loads(content),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "usage": completion.usage.model_dump() if completion.usage else None,
    }


def normalize_plan(
    raw_plan: dict[str, Any], requirements: dict[str, Any]
) -> dict[str, Any]:
    """Enforce user-selected metadata and safe, unique field names."""

    fields: list[dict[str, str]] = []
    used_names: set[str] = set()
    for index, raw_field in enumerate(raw_plan.get("fields", []), start=1):
        field_type = raw_field.get("type")
        if field_type not in SUPPORTED_FIELD_TYPES:
            continue

        base_name = sanitize_name(
            raw_field.get("name", ""),
            f"field_{index}",
            prefix_leading_digit=True,
        )
        unique_name = base_name
        suffix = 2
        while unique_name in used_names:
            unique_name = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(unique_name)

        fields.append(
            {
                "name": unique_name,
                "type": field_type,
                "description": str(raw_field.get("description", "")).strip(),
            }
        )

    if not fields:
        raise ValueError("The model did not propose any supported fields.")

    # User-selected format and counts are authoritative. The model contributes
    # field design and generation instructions but cannot silently change the
    # requested output size or type.
    return {
        "dataset_name": requirements["dataset_name"],
        "purpose": requirements["purpose"],
        "format": requirements["format"],
        "record_count": requirements["record_count"],
        "file_count": requirements["file_count"],
        "fields": fields,
        "generation_instructions": str(
            raw_plan.get("generation_instructions", "")
        ).strip(),
    }


# Step 6: Show each plan and collaborate until the user approves or quits.
def review_plan(
    client: OpenAI, requirements: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return an approved plan and metadata for every planning request."""

    revision_notes: list[str] = []
    planning_runs: list[dict[str, Any]] = []

    while True:
        model_run = request_plan(client, requirements, revision_notes)
        plan = normalize_plan(model_run["plan"], requirements)
        planning_runs.append(
            {
                "raw_plan": model_run["plan"],
                "normalized_plan": plan,
                "revision_notes": list(revision_notes),
                "elapsed_seconds": model_run["elapsed_seconds"],
                "usage": model_run["usage"],
            }
        )

        print("\nProposed dataset plan:\n")
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        decision = read_text(
            "Enter 'approve', 'quit', or describe a revision", "approve"
        )
        if decision.lower() == "approve":
            return plan, planning_runs
        if decision.lower() == "quit":
            raise UserCancelled

        revision_notes.append(decision)
        print("\nAsking Ornith to revise the plan...")


# Step 7: Build a record schema dynamically from the approved field plan.
def field_json_schema(field: dict[str, str]) -> dict[str, Any]:
    """Translate one teaching field type into a JSON-schema property."""

    field_type = field["type"]
    if field_type == "date":
        return {
            "type": "string",
            "description": f"ISO date YYYY-MM-DD. {field['description']}",
        }
    return {"type": field_type, "description": field["description"]}


def build_records_response_format(plan: dict[str, Any]) -> dict[str, Any]:
    """Return a schema requiring exact keys and the requested record count."""

    properties = {
        field["name"]: field_json_schema(field) for field in plan["fields"]
    }
    required_names = [field["name"] for field in plan["fields"]]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"{plan['dataset_name']}_records",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "records": {
                        "type": "array",
                        "minItems": plan["record_count"],
                        "maxItems": plan["record_count"],
                        "items": {
                            "type": "object",
                            "properties": properties,
                            "required": required_names,
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["records"],
                "additionalProperties": False,
            },
        },
    }


# Step 8: Ask Ornith for records matching the approved dynamic schema.
def generate_records(client: OpenAI, plan: dict[str, Any]) -> dict[str, Any]:
    """Return generated records with timing and token metadata."""

    prompt = (
        f"Generate exactly {plan['record_count']} fully fictional records for "
        f"this approved dataset plan:\n{json.dumps(plan, indent=2)}\n\n"
        "Follow every field description and generation instruction. Create a "
        "useful mix of normal, boundary, and edge-case values. Keep values "
        "internally consistent. Do not use real names, addresses, phone numbers, "
        "account identifiers, or other real personal information."
    )

    started = perf_counter()
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You generate compact, fully fictional test records.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=build_records_response_format(plan),
        temperature=0.6,
    )
    elapsed_seconds = perf_counter() - started

    content = completion.choices[0].message.content or "{}"
    parsed = json.loads(content)
    return {
        "records": parsed.get("records", []),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "usage": completion.usage.model_dump() if completion.usage else None,
    }


# Step 9: Validate every key, scalar type, and date before writing files.
def value_matches_type(value: Any, field_type: str) -> bool:
    """Return whether one generated scalar matches the approved field type."""

    if field_type == "string":
        return isinstance(value, str)
    if field_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if field_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if field_type == "boolean":
        return isinstance(value, bool)
    if field_type == "date":
        if not isinstance(value, str):
            return False
        try:
            date.fromisoformat(value)
            return True
        except ValueError:
            return False
    return False


def validate_records(
    records: list[dict[str, Any]], plan: dict[str, Any]
) -> list[str]:
    """Return human-readable validation errors; an empty list means success."""

    errors: list[str] = []
    expected_names = {field["name"] for field in plan["fields"]}
    field_types = {field["name"]: field["type"] for field in plan["fields"]}

    if len(records) != plan["record_count"]:
        errors.append(
            f"Expected {plan['record_count']} records but received {len(records)}."
        )

    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            errors.append(f"Record {index} is not an object.")
            continue
        if set(record) != expected_names:
            errors.append(f"Record {index} does not contain the exact field set.")
            continue
        for field_name, field_type in field_types.items():
            if not value_matches_type(record[field_name], field_type):
                errors.append(
                    f"Record {index} field {field_name!r} is not {field_type}."
                )

    return errors


# Step 10: Split records as evenly as possible across the requested files.
def split_records(
    records: list[dict[str, Any]], file_count: int
) -> list[list[dict[str, Any]]]:
    """Return balanced record groups without dropping or duplicating rows."""

    base_size, extra = divmod(len(records), file_count)
    groups: list[list[dict[str, Any]]] = []
    start = 0
    for index in range(file_count):
        group_size = base_size + (1 if index < extra else 0)
        groups.append(records[start : start + group_size])
        start += group_size
    return groups


# Step 11: Serialize each group using trusted standard-library writers.
def output_filename(dataset_name: str, output_format: str, index: int, total: int) -> str:
    """Return a readable file name for single-file or multi-file output."""

    if total == 1:
        return f"{dataset_name}.{output_format}"
    return f"{dataset_name}_{index:03d}.{output_format}"


def write_csv_file(
    path: Path, records: list[dict[str, Any]], field_names: list[str]
) -> None:
    """Write records with one header row using the standard CSV module."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(records)


def write_json_file(path: Path, records: list[dict[str, Any]]) -> None:
    """Write one indented JSON array."""

    path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_xml_file(path: Path, records: list[dict[str, Any]]) -> None:
    """Write records beneath a dataset root with one element per field."""

    root = ET.Element("dataset")
    for record in records:
        record_element = ET.SubElement(root, "record")
        for field_name, value in record.items():
            field_element = ET.SubElement(record_element, field_name)
            if isinstance(value, bool):
                field_element.text = str(value).lower()
            else:
                field_element.text = str(value)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def write_dataset_files(
    output_directory: Path,
    plan: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[Path]:
    """Split and write all requested dataset files."""

    output_directory.mkdir(parents=True, exist_ok=False)
    field_names = [field["name"] for field in plan["fields"]]
    groups = split_records(records, plan["file_count"])
    written_paths: list[Path] = []

    for index, group in enumerate(groups, start=1):
        filename = output_filename(
            plan["dataset_name"], plan["format"], index, len(groups)
        )
        path = output_directory / filename
        if plan["format"] == "csv":
            write_csv_file(path, group, field_names)
        elif plan["format"] == "json":
            write_json_file(path, group)
        else:
            write_xml_file(path, group)
        written_paths.append(path)

    return written_paths


# Step 12: Coordinate the workflow, write a manifest, and report completion.
def run_app() -> dict[str, Any]:
    """Run the interactive design, approval, generation, and writing workflow."""

    client = create_client()
    requirements = collect_requirements()
    plan, planning_runs = review_plan(client, requirements)

    print("\nGenerating records with Ornith...")
    generation = generate_records(client, plan)
    validation_errors = validate_records(generation["records"], plan)
    if validation_errors:
        raise ValueError("Record validation failed: " + " | ".join(validation_errors))

    # A timestamp gives normal interactive runs their own directory. Tests or
    # automation can set DATASET_RUN_ID for a stable, descriptive suffix.
    run_id = sanitize_name(
        os.getenv("DATASET_RUN_ID", datetime.now().strftime("%Y%m%d_%H%M%S")),
        "run",
    )
    output_directory = OUTPUT_ROOT / f"{plan['dataset_name']}_{run_id}"
    written_paths = write_dataset_files(
        output_directory, plan, generation["records"]
    )

    manifest = {
        "exercise": "interactive_fake_dataset_creator",
        "backend": "local_lm_studio",
        "model": MODEL,
        "base_url": BASE_URL,
        "requirements": requirements,
        "approved_plan": plan,
        "planning_runs": planning_runs,
        "generation": {
            "elapsed_seconds": generation["elapsed_seconds"],
            "usage": generation["usage"],
            "record_count": len(generation["records"]),
            "validation_passed": True,
            "validation_errors": [],
        },
        "output_files": [str(path) for path in written_paths],
    }
    manifest_path = output_directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nDataset created successfully.")
    print(f"Output directory: {output_directory}")
    print(f"Records: {len(generation['records'])}")
    print(f"Files: {len(written_paths)}")
    for path in written_paths:
        print(f"  - {path}")
    print(f"Manifest: {manifest_path}")
    return manifest


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    try:
        run_app()
    except UserCancelled:
        print("\nDataset creation cancelled.")
    except (OpenAIError, json.JSONDecodeError, ValueError, OSError) as error:
        print(f"\nDataset creation failed: {error}")
        raise SystemExit(1) from error
