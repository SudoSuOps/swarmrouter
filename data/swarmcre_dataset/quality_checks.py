"""
SwarmCRE Dataset Factory -- Quality Pipeline

6-gate quality pipeline for validating generated JSONL training records.
Each gate is a standalone check; the pipeline runs all gates and aggregates
results.  A record must pass every gate to be emitted.

Gates:
    1. JSONValidityGate      -- answer is valid JSON when output_schema is set
    2. PydanticSchemaGate    -- JSON answer validates against the Pydantic model
    3. NumericVerificationGate -- gold numeric targets appear in the answer text
    4. ConceptPresenceGate   -- answer contains expected CRE domain terms
    5. DuplicateDetectionGate -- no duplicate questions (MD5 fingerprinting)
    6. OutputLengthGate      -- answer meets minimum length thresholds
"""

import json
import hashlib
import re

from .schemas import SCHEMA_REGISTRY

try:
    from pydantic import ValidationError
except ImportError:  # pragma: no cover
    ValidationError = Exception


# =====================================================================
# CRE concept terms expected per task type
# =====================================================================

_CONCEPT_TERMS_BY_TASK = {
    "underwriting_calc": [
        "NOI", "cap", "DSCR", "LTV", "debt", "EGI",
        "vacancy", "/SF", "yield", "cash", "PGI", "value",
    ],
    "rent_roll_extraction": [
        "tenant", "SF", "rent", "lease", "vacancy", "WALT",
    ],
    "t12_normalization": [
        "NOI", "revenue", "expense", "EGI", "management fee",
        "property tax", "insurance",
    ],
    "lease_abstract_extraction": [
        "tenant", "lease", "rent", "term", "escalation", "option",
    ],
    "ic_memo": [
        "NOI", "cap rate", "recommendation", "risk", "tenant",
        "market", "DSCR",
    ],
    "lease_reasoning": [
        "tenant", "rent", "lease", "market", "escalation", "renewal",
    ],
    "market_comp_narrative": [
        "comp", "/SF", "cap", "value", "price",
    ],
    "risk_triage": [
        "risk", "recommendation", "severity", "mitigation",
    ],
    "loi_deliverable": [
        "purchase price", "earnest money", "due diligence", "closing",
    ],
    "exchange_1031": [
        "1031", "exchange", "basis", "gain", "boot",
        "deferred", "replacement", "relinquished",
        "45", "180", "intermediary",
    ],
    "tax_analysis": [
        "depreciation", "basis", "gain", "tax",
        "recapture", "1250", "cost segregation",
        "capital", "rate", "deduction",
    ],
    "structured_agent_output": [
        "confidence", "reasoning", "output",
    ],
}

# Minimum number of concept terms that must appear (per task type).
# Default: at least 2 terms from the task's list must be present.
_MIN_CONCEPT_HITS = 2


# =====================================================================
# Base class
# =====================================================================


class QualityGate:
    """Base class for a single quality gate.

    Subclasses implement ``check(record)`` which returns
    ``(passed: bool, reason: str)``.  When the gate passes, reason is
    the empty string.
    """

    name: str = "base"

    def check(self, record: dict) -> tuple[bool, str]:
        """Validate a single record.

        Args:
            record: A fully-assembled JSONL training record with keys
                ``messages``, ``metadata``, and ``gold``.

        Returns:
            (True, "") on pass, (False, failure_reason) on fail.
        """
        raise NotImplementedError


# =====================================================================
# Gate 1: JSON validity
# =====================================================================


class JSONValidityGate(QualityGate):
    """Checks that the assistant answer is valid JSON when an output_schema
    is specified in the gold block.

    Records without an output_schema pass automatically (they are
    free-text answers, not structured extraction tasks).
    """

    name = "json_validity"

    def check(self, record: dict) -> tuple[bool, str]:
        output_schema = record.get("gold", {}).get("output_schema")
        if not output_schema:
            return True, ""

        answer = _extract_answer(record)
        try:
            json.loads(answer)
            return True, ""
        except (json.JSONDecodeError, TypeError) as exc:
            return False, f"JSONValidityGate: invalid JSON in answer -- {exc}"


# =====================================================================
# Gate 2: Pydantic schema validation
# =====================================================================


class PydanticSchemaGate(QualityGate):
    """Validates the parsed JSON answer against the Pydantic model
    registered in ``SCHEMA_REGISTRY``.

    Passes automatically when no output_schema is set or the schema
    name is not found in the registry (to avoid blocking on newly added
    task types before schemas are registered).
    """

    name = "pydantic_schema"

    def check(self, record: dict) -> tuple[bool, str]:
        schema_name = record.get("gold", {}).get("output_schema")
        if not schema_name:
            return True, ""

        model_cls = SCHEMA_REGISTRY.get(schema_name)
        if model_cls is None:
            # Schema not registered -- cannot validate; let it pass.
            return True, ""

        answer = _extract_answer(record)
        try:
            data = json.loads(answer)
        except (json.JSONDecodeError, TypeError):
            # JSON parsing failure is caught by JSONValidityGate; don't
            # double-report here.
            return True, ""

        try:
            model_cls.model_validate(data)
            return True, ""
        except ValidationError as exc:
            error_count = exc.error_count() if hasattr(exc, "error_count") else len(exc.errors()) if hasattr(exc, "errors") else 1
            return False, f"PydanticSchemaGate: {error_count} validation error(s) against {schema_name}"


# =====================================================================
# Gate 3: Numeric verification
# =====================================================================


# Precompiled patterns for extracting numbers from text.
# Matches integers, decimals, and dollar-formatted numbers.
_NUMBER_PATTERN = re.compile(
    r"""
    (?<![a-zA-Z])          # not preceded by a letter (avoids partial words)
    [\$]?                  # optional dollar sign
    -?                     # optional negative sign
    (?:\d{1,3}(?:,\d{3})+  # comma-separated groups (must have at least one comma)
       |\d+)               # or plain digits  e.g. 2000, 26840
    (?:\.\d+)?             # optional decimal part
    [%x]?                  # optional percent sign or 'x' (DSCR notation)
    (?![a-zA-Z])           # not followed by another letter
    """,
    re.VERBOSE,
)


def _parse_number(s: str) -> float | None:
    """Parse a single number token into a float."""
    s = s.strip().replace("$", "").replace(",", "").replace("%", "").rstrip("x")
    try:
        return float(s)
    except ValueError:
        return None


def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from a text string."""
    matches = _NUMBER_PATTERN.findall(text)
    numbers = []
    for m in matches:
        val = _parse_number(m)
        if val is not None:
            numbers.append(val)
    return numbers


class NumericVerificationGate(QualityGate):
    """Checks that gold.numeric_targets values appear in the answer text.

    Tolerance:
        - Money fields (values >= 100 or field names containing common money
          indicators): +/- $1
        - Ratio / percentage fields: +/- 0.01

    A record passes when *all* gold numeric targets are found within
    tolerance in the answer text.  Records with no numeric targets pass
    automatically.
    """

    name = "numeric_verification"

    _MONEY_FIELDS = frozenset({
        "pgi", "egi", "noi", "value", "total_opex", "management_fee",
        "vacancy_loss", "loan_amount", "annual_debt_service", "equity",
        "purchase_price", "earnest_money", "annual_rent", "base_rent",
        "cam_reimbursements", "other_income", "potential_gross_income",
        "effective_gross_income", "sale_price",
    })

    # Fields stored as decimals (0.065) but often shown as percentages (6.5%)
    _RATE_FIELDS = frozenset({
        "cap_rate", "vacancy_rate", "ltv", "debt_yield", "cash_on_cash",
        "escalation_rate", "earnest_money_pct",
    })

    def _is_money_field(self, field_name: str, value: float) -> bool:
        """Heuristic: is this a dollar-denominated field?"""
        if field_name in self._MONEY_FIELDS:
            return True
        # Large absolute values are likely dollar amounts
        if abs(value) >= 100:
            return True
        return False

    def check(self, record: dict) -> tuple[bool, str]:
        numeric_targets = record.get("gold", {}).get("numeric_targets", {})
        if not numeric_targets:
            return True, ""

        answer = _extract_answer(record)
        answer_numbers = _extract_numbers(answer)

        if not answer_numbers:
            return False, (
                f"NumericVerificationGate: answer contains no numbers but "
                f"{len(numeric_targets)} target(s) expected"
            )

        missing = []
        for field, expected in numeric_targets.items():
            if expected is None:
                continue
            # Skip nested dicts / lists (e.g. sensitivity tables)
            if isinstance(expected, (dict, list)):
                continue

            expected_f = float(expected)
            tolerance = 1.0 if self._is_money_field(field, expected_f) else 0.01

            found = False
            for ans_num in answer_numbers:
                if abs(ans_num - expected_f) <= tolerance:
                    found = True
                    break

            # For rate fields, also check if value * 100 appears (6.5 for 0.065)
            if not found and field in self._RATE_FIELDS and 0 < abs(expected_f) < 1:
                pct_value = expected_f * 100
                for ans_num in answer_numbers:
                    if abs(ans_num - pct_value) <= 0.1:
                        found = True
                        break

            if not found:
                missing.append(f"{field}={expected}")

        if missing:
            return False, (
                f"NumericVerificationGate: missing targets in answer: "
                f"{', '.join(missing[:5])}"
                + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else "")
            )

        return True, ""


# =====================================================================
# Gate 4: Concept presence
# =====================================================================


class ConceptPresenceGate(QualityGate):
    """Checks that the answer contains key CRE domain terms appropriate
    for the task type.

    Uses case-insensitive regex matching so that "NOI" matches "noi",
    "Net Operating Income", etc.  At least ``_MIN_CONCEPT_HITS`` terms
    from the task's concept list must appear.
    """

    name = "concept_presence"

    def check(self, record: dict) -> tuple[bool, str]:
        task_type = record.get("metadata", {}).get("task_type", "")
        terms = _CONCEPT_TERMS_BY_TASK.get(task_type)
        if not terms:
            # Unknown task type -- no concept list; pass.
            return True, ""

        answer = _extract_answer(record)
        answer_lower = answer.lower()

        hits = 0
        for term in terms:
            # Use regex for terms with dots (e.g. "cash.on.cash" matches
            # "cash-on-cash", "cash on cash", etc.)
            pattern = term.replace(".", r"[\s\-]?")
            if re.search(pattern, answer_lower, re.IGNORECASE):
                hits += 1

        if hits < _MIN_CONCEPT_HITS:
            return False, (
                f"ConceptPresenceGate: only {hits}/{_MIN_CONCEPT_HITS} required "
                f"CRE terms found for task_type={task_type}"
            )

        return True, ""


# =====================================================================
# Gate 5: Duplicate detection
# =====================================================================


class DuplicateDetectionGate(QualityGate):
    """Tracks MD5 fingerprints of user questions and rejects duplicates.

    The fingerprint is computed over the normalized (lowered, whitespace-
    collapsed) question text.  This gate is stateful and must be
    instantiated once per pipeline run.
    """

    name = "duplicate_detection"

    def __init__(self):
        self._seen: set[str] = set()

    def check(self, record: dict) -> tuple[bool, str]:
        question = _extract_question(record)
        # Include record ID in fingerprint so same-deal different-slot records
        # are not falsely flagged as duplicates
        record_id = record.get("id", "")
        fingerprint = self._fingerprint(f"{record_id}:{question}")

        if fingerprint in self._seen:
            return False, (
                f"DuplicateDetectionGate: duplicate question detected "
                f"(md5={fingerprint[:12]}...)"
            )

        self._seen.add(fingerprint)
        return True, ""

    @staticmethod
    def _fingerprint(text: str) -> str:
        """MD5 hash of normalized text."""
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        return hashlib.md5(normalized.encode("utf-8")).hexdigest()

    def reset(self):
        """Clear the fingerprint cache (e.g. between shards)."""
        self._seen.clear()


# =====================================================================
# Gate 6: Output length
# =====================================================================


class OutputLengthGate(QualityGate):
    """Ensures the assistant answer meets minimum length thresholds.

    Thresholds:
        - JSON tasks (output_schema is set): at least 20 characters
        - Free-text tasks: at least 50 characters
    """

    name = "output_length"

    _MIN_LEN_JSON = 20
    _MIN_LEN_TEXT = 50

    def check(self, record: dict) -> tuple[bool, str]:
        answer = _extract_answer(record)
        output_schema = record.get("gold", {}).get("output_schema")

        if output_schema:
            min_len = self._MIN_LEN_JSON
            task_kind = "JSON"
        else:
            min_len = self._MIN_LEN_TEXT
            task_kind = "text"

        if len(answer) < min_len:
            return False, (
                f"OutputLengthGate: {task_kind} answer too short "
                f"({len(answer)} < {min_len} chars)"
            )

        return True, ""


# =====================================================================
# Aggregated pipeline
# =====================================================================


class QualityPipeline:
    """Runs all quality gates on a record and aggregates results.

    Usage::

        pipeline = QualityPipeline()
        for record in records:
            passed, failures = pipeline.check(record)
            if passed:
                emit(record)
            else:
                log_failures(record, failures)
        print(pipeline.stats)
    """

    def __init__(self):
        self._gates: list[QualityGate] = [
            JSONValidityGate(),
            PydanticSchemaGate(),
            NumericVerificationGate(),
            ConceptPresenceGate(),
            DuplicateDetectionGate(),
            OutputLengthGate(),
        ]
        self._pass_counts: dict[str, int] = {g.name: 0 for g in self._gates}
        self._fail_counts: dict[str, int] = {g.name: 0 for g in self._gates}
        self._total_checked: int = 0

    def check(self, record: dict) -> tuple[bool, list[str]]:
        """Run all gates on a record.

        Args:
            record: A fully-assembled JSONL training record.

        Returns:
            (all_passed, list_of_failure_reasons).
            ``all_passed`` is True only when every gate passes.
            ``list_of_failure_reasons`` is empty on full pass.
        """
        self._total_checked += 1
        failures = []

        for gate in self._gates:
            passed, reason = gate.check(record)
            if passed:
                self._pass_counts[gate.name] += 1
            else:
                self._fail_counts[gate.name] += 1
                failures.append(reason)

        return len(failures) == 0, failures

    @property
    def stats(self) -> dict:
        """Return pass/fail counts per gate and overall totals.

        Returns:
            Dict of the form::

                {
                    "total_checked": int,
                    "gates": {
                        "gate_name": {"pass": int, "fail": int},
                        ...
                    }
                }
        """
        gate_stats = {}
        for gate in self._gates:
            gate_stats[gate.name] = {
                "pass": self._pass_counts[gate.name],
                "fail": self._fail_counts[gate.name],
            }
        return {
            "total_checked": self._total_checked,
            "gates": gate_stats,
        }

    def reset(self):
        """Reset all counters and stateful gates."""
        self._pass_counts = {g.name: 0 for g in self._gates}
        self._fail_counts = {g.name: 0 for g in self._gates}
        self._total_checked = 0
        for gate in self._gates:
            if hasattr(gate, "reset"):
                gate.reset()


# =====================================================================
# Shared helpers
# =====================================================================


def _extract_answer(record: dict) -> str:
    """Extract the assistant answer text from a record."""
    messages = record.get("messages", [])
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def _extract_question(record: dict) -> str:
    """Extract the user question text from a record."""
    messages = record.get("messages", [])
    for msg in messages:
        if msg.get("role") == "user":
            return msg.get("content", "")
    return ""
