"""Shared prompt construction for restaurant response generation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


RESPONSE_INSTRUCTION = "Generate a short grounded restaurant assistant response using only the provided evidence."

PUBLIC_EVIDENCE_FIELDS = ("name", "food", "area", "pricerange", "address", "postcode", "phone", "type")
STATE_FIELDS = (
    "food",
    "area",
    "pricerange",
    "day",
    "time",
    "people",
    "booking_status",
    "booking_reference",
    "restaurant",
    "status",
    "second_booking_reference",
    "supported_cuisines",
)


@dataclass(frozen=True)
class ResponsePromptFields:
    intent: str
    user: str
    state: dict[str, str]
    evidence_records: list[dict[str, str]]
    missing_slots: list[str]


def public_evidence_record(record: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return only evidence fields that are safe to expose in prompts."""

    if not record:
        return {}
    return {
        key: record.get(key)
        for key in PUBLIC_EVIDENCE_FIELDS
        if record.get(key) not in (None, "", [])
    }


def format_evidence(records: Iterable[Mapping[str, Any]] | None) -> str:
    chunks: list[str] = []
    for record in records or []:
        public = public_evidence_record(record)
        fields = [
            f"{key}={public[key]}"
            for key in PUBLIC_EVIDENCE_FIELDS
            if key in public
        ]
        if fields:
            chunks.append("; ".join(fields))
    return " | ".join(chunks)


def format_state(state: Mapping[str, Any] | str | None) -> str:
    if state is None:
        return ""
    if isinstance(state, str):
        return " ".join(state.split())

    keys = [key for key in STATE_FIELDS if key in state]
    keys.extend(sorted(key for key in state if key not in STATE_FIELDS))
    parts: list[str] = []
    for key in keys:
        value = state.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, (list, tuple, set)):
            value = ", ".join(str(item) for item in value if item not in (None, ""))
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def build_response_input(
    *,
    intent: str,
    user: str,
    state: Mapping[str, Any] | str | None = None,
    evidence_records: Iterable[Mapping[str, Any]] | None = None,
    missing_slots: Iterable[str] | None = None,
) -> str:
    """Build the model input text shared by data, training, runtime and eval."""

    parts = [
        "Task: Generate a grounded restaurant assistant response.",
        f"Intent: {intent}",
        f"User: {user}",
    ]
    state_text = format_state(state)
    evidence_text = format_evidence(evidence_records)
    missing = [str(slot) for slot in missing_slots or [] if str(slot).strip()]
    if state_text:
        parts.append(f"State: {state_text}")
    if evidence_text:
        parts.append(f"Evidence: {evidence_text}")
    if missing:
        parts.append("Missing slots: " + json.dumps(missing, ensure_ascii=False))
    parts.append("Response:")
    return "\n".join(parts)


def build_response_prompt(
    *,
    intent: str,
    user: str,
    state: Mapping[str, Any] | str | None = None,
    evidence_records: Iterable[Mapping[str, Any]] | None = None,
    missing_slots: Iterable[str] | None = None,
    instruction: str = RESPONSE_INSTRUCTION,
) -> str:
    return instruction + "\n" + build_response_input(
        intent=intent,
        user=user,
        state=state,
        evidence_records=evidence_records,
        missing_slots=missing_slots,
    )


def prompt_from_row(row: Mapping[str, Any]) -> str:
    """Return the exact model prompt for a dataset row without target leakage."""

    instruction = str(row.get("instruction") or RESPONSE_INSTRUCTION).strip()
    input_text = str(row.get("input") or "").strip()
    if not instruction or not input_text:
        raise ValueError("Rows must contain non-empty instruction and input fields.")
    return instruction + "\n" + input_text


def parse_response_input(input_text: str) -> ResponsePromptFields:
    intent = "search"
    user = ""
    state: dict[str, str] = {}
    evidence_records: list[dict[str, str]] = []
    missing_slots: list[str] = []

    for line in str(input_text or "").splitlines():
        if line.startswith("Intent:"):
            intent = line.split(":", 1)[1].strip() or intent
        elif line.startswith("User:"):
            user = line.split(":", 1)[1].strip()
        elif line.startswith("State:"):
            state = parse_state_text(line.split(":", 1)[1].strip())
        elif line.startswith("Evidence:"):
            evidence_records = parse_evidence_text(line.split(":", 1)[1].strip())
        elif line.startswith("Missing slots:"):
            missing_slots = parse_missing_slots(line.split(":", 1)[1].strip())

    return ResponsePromptFields(
        intent=intent,
        user=user,
        state=state,
        evidence_records=evidence_records,
        missing_slots=missing_slots,
    )


def parse_state_text(state_text: str) -> dict[str, str]:
    state: dict[str, str] = {}
    for part in re.split(r",\s+(?=[A-Za-z_][A-Za-z0-9_]*=)", state_text):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            state[key] = value
    return state


def parse_evidence_text(evidence_text: str) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for chunk in evidence_text.split("|"):
        record: dict[str, str] = {}
        for part in chunk.split(";"):
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                record[key] = value
        if record:
            records.append(record)
    return records


def parse_missing_slots(raw: str) -> list[str]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed if str(item).strip()]
