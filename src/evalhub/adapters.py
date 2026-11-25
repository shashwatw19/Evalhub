"""
Input adapters that normalize agent records into EvalHub's canonical case format.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Set

from evalhub.models import EvaluationCase


RecordAdapter = Callable[[Dict[str, Any]], EvaluationCase]


def _parse_if_serialized(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    return value


class GenericAdapter:
    """Adapt a canonical agent-run record, preserving unknown fields as metadata."""

    @staticmethod
    def adapt(record: Dict[str, Any]) -> EvaluationCase:
        known_fields = set(EvaluationCase.model_fields)
        data = {
            key: _parse_if_serialized(value)
            for key, value in record.items()
            if key in known_fields
        }

        supplied_metadata = data.get("metadata") or {}
        if not isinstance(supplied_metadata, dict):
            raise ValueError("The 'metadata' field must be a JSON object.")

        extra_metadata = {
            key: value for key, value in record.items() if key not in known_fields
        }
        data["metadata"] = {**extra_metadata, **supplied_metadata}
        return EvaluationCase.model_validate(data)


ADAPTER_REGISTRY: Dict[str, RecordAdapter] = {
    "generic": GenericAdapter.adapt,
}


def register_adapter(name: str, adapter: RecordAdapter) -> None:
    """Register a Python adapter for a source format not using the canonical schema."""
    normalized_name = name.strip()
    if not normalized_name:
        raise ValueError("Adapter name must be a non-empty string.")
    if normalized_name in ADAPTER_REGISTRY:
        raise ValueError(f"Adapter '{normalized_name}' is already registered.")
    ADAPTER_REGISTRY[normalized_name] = adapter


def get_adapter(name: str) -> RecordAdapter:
    """Retrieve a registered record adapter."""
    if name not in ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown adapter '{name}'. Available adapters: {sorted(ADAPTER_REGISTRY)}"
        )
    return ADAPTER_REGISTRY[name]


def normalize_records(
    records: List[Dict[str, Any]], adapter_name: str = "generic"
) -> List[EvaluationCase]:
    """Normalize records and enforce unique case IDs within the dataset."""
    adapter = get_adapter(adapter_name)
    seen_ids: Set[str] = set()
    normalized: List[EvaluationCase] = []

    for idx, raw in enumerate(records, start=1):
        case = adapter(raw)
        if case.case_id in seen_ids:
            raise ValueError(
                f"Duplicate case_id '{case.case_id}' detected at record {idx}. "
                "Dataset case IDs must be unique within a single run."
            )
        seen_ids.add(case.case_id)
        normalized.append(case)

    return normalized
