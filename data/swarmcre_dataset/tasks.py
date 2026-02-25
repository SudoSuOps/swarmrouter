"""
SwarmCRE Dataset Factory -- Task Builder

Wires deal skeletons to templates to produce JSONL-ready training records.
Each deal produces exactly 10 task records with deterministic distribution
across task types and difficulty levels.

Determinism: SHA-256 per-deal seeding ensures identical output for the same
deal_id regardless of generation order or parallelism.
"""

import hashlib
import random

from .constants import TASK_DISTRIBUTION, DIFFICULTY_WEIGHTS, TASK_SYSTEM_PROMPTS


# =====================================================================
# Per-deal task allocation: 10 tasks per deal
# Maps task_type -> count per deal
# =====================================================================

TASKS_PER_DEAL = {
    "underwriting_calc": 3,
    "rent_roll_extraction": 1,
    "t12_normalization": 1,
    "lease_abstract_extraction": 1,
    "ic_memo": 2,
    "lease_reasoning": 1,
    "market_comp_narrative": 1,
}

# Asset types that cause specific task types to be probabilistically skipped.
# Key = task_type, value = set of asset_types where the task is unsuitable.
_SKIP_RULES = {
    "rent_roll_extraction": {"industrial_land"},
    "t12_normalization": {"industrial_land", "ios_truck_yard"},
    "lease_abstract_extraction": {"industrial_land"},
    "lease_reasoning": {"industrial_land"},
}

# Replacement task types used when a primary task is skipped for a deal.
# Falls back through the list until one is compatible.
_FALLBACK_TASKS = [
    "underwriting_calc",
    "ic_memo",
    "market_comp_narrative",
    "lease_reasoning",
]


class TaskBuilder:
    """Converts deal skeletons into JSONL training records via templates.

    Each deal produces 10 task records drawn from a fixed allocation across
    task types.  Templates provide the question/answer text, output schemas,
    and gold-field specifications.  The builder handles difficulty selection,
    skip logic for incompatible asset/task combinations, and deterministic
    RNG seeding.

    Args:
        templates: A TemplateRegistry instance that exposes:
            - get_templates(task_type, difficulty) -> list of template objects
            Each template object must have attributes:
                template_id: str
                difficulty: str  ("low", "medium", "high")
                gold_fields: list[str]
                output_schema: str | None
                requires_enrichment: bool
            And a render(deal_dict) -> (question: str, answer: str) method.
    """

    def __init__(self, templates):
        self._templates = templates
        self._difficulty_levels = list(DIFFICULTY_WEIGHTS.keys())
        self._difficulty_weights = list(DIFFICULTY_WEIGHTS.values())

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _deal_rng(deal_id: str) -> random.Random:
        """Create a deterministic RNG seeded by the deal_id."""
        h = hashlib.sha256(f"tasks:{deal_id}".encode()).hexdigest()[:16]
        return random.Random(h)

    def _pick_difficulty(self, rng: random.Random) -> str:
        """Weighted random difficulty: 20% low, 55% medium, 25% high."""
        r = rng.random()
        cumulative = 0.0
        for level, weight in zip(self._difficulty_levels, self._difficulty_weights):
            cumulative += weight
            if r <= cumulative:
                return level
        return self._difficulty_levels[-1]

    def _pick_template(self, rng: random.Random, task_type: str, difficulty: str):
        """Select a template for the given task_type and difficulty.

        If no templates exist at the requested difficulty, tries adjacent
        difficulties (medium -> low -> high) so generation never stalls.

        Returns:
            A template object, or None if no templates exist for this task type.
        """
        # Primary lookup
        candidates = self._templates.get_templates(task_type, difficulty)
        if candidates:
            return rng.choice(candidates)

        # Fallback: try other difficulties
        fallback_order = ["medium", "low", "high"]
        for alt in fallback_order:
            if alt == difficulty:
                continue
            candidates = self._templates.get_templates(task_type, alt)
            if candidates:
                return rng.choice(candidates)

        return None

    def _should_skip(self, rng: random.Random, task_type: str, deal_dict: dict) -> bool:
        """Determine whether a task type should be skipped for this deal.

        Skipping is deterministic for hard incompatibilities (land deals have
        no rent roll) and probabilistic for soft mismatches (IOS deals skip
        T-12 80% of the time).
        """
        skip_assets = _SKIP_RULES.get(task_type)
        if skip_assets is None:
            return False

        asset_type = deal_dict.get("asset_type", "")
        if asset_type not in skip_assets:
            return False

        # Hard skip for land (no building, no tenants, no financials)
        if asset_type == "industrial_land":
            return True

        # Soft skip for IOS on T-12 (80% chance)
        if asset_type == "ios_truck_yard" and task_type == "t12_normalization":
            return rng.random() < 0.80

        return True

    def _find_replacement(self, rng: random.Random, task_type: str, deal_dict: dict) -> str:
        """Find a replacement task type when the primary one is skipped."""
        for fallback in _FALLBACK_TASKS:
            if fallback == task_type:
                continue
            if not self._should_skip(rng, fallback, deal_dict):
                return fallback
        # Last resort: always-safe task type
        return "market_comp_narrative"

    def _build_record(
        self,
        deal_dict: dict,
        task_type: str,
        template,
        question: str,
        answer: str,
    ) -> dict:
        """Assemble a single JSONL training record."""
        system_prompt = TASK_SYSTEM_PROMPTS.get(task_type, TASK_SYSTEM_PROMPTS.get("underwriting_calc", ""))

        # Extract numeric gold targets that this template cares about
        deal_gold = deal_dict.get("gold", {})
        numeric_targets = {}
        for field in template.gold_fields:
            if field in deal_gold:
                numeric_targets[field] = deal_gold[field]

        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "metadata": {
                "deal_id": deal_dict["deal_id"],
                "task_type": task_type,
                "template_id": template.template_id,
                "difficulty": template.difficulty,
                "asset_type": deal_dict["asset_type"],
                "market_tier": deal_dict["market_tier"],
                "market_name": deal_dict["market_name"],
                "sf": deal_dict["sf"],
                "lease_type": deal_dict.get("lease_type", ""),
                "requires_enrichment": template.requires_enrichment,
            },
            "gold": {
                "numeric_targets": numeric_targets,
                "output_schema": template.output_schema,
            },
        }

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    def build_tasks_for_deal(self, deal_dict: dict, deal_idx: int) -> list[dict]:
        """Generate 10 task records for a single deal skeleton.

        The task allocation follows TASKS_PER_DEAL (3 underwriting_calc,
        2 ic_memo, 1 each of the rest = 10 total).  Task types that are
        incompatible with the deal's asset type are replaced with fallback
        types so the count always reaches 10.

        Args:
            deal_dict: Serialized deal skeleton (from DealSkeleton.to_dict()).
            deal_idx: Ordinal index of this deal in the generation run.

        Returns:
            List of 10 record dicts, each ready for JSONL serialization.
        """
        rng = self._deal_rng(deal_dict["deal_id"])
        records = []

        # Build the task manifest: expand TASKS_PER_DEAL into an ordered list
        task_manifest = []
        for task_type, count in TASKS_PER_DEAL.items():
            for _ in range(count):
                task_manifest.append(task_type)

        # Shuffle for variety within each deal (deterministic via rng)
        rng.shuffle(task_manifest)

        for slot_idx, task_type in enumerate(task_manifest):
            # Check skip rules -- replace if incompatible
            if self._should_skip(rng, task_type, deal_dict):
                task_type = self._find_replacement(rng, task_type, deal_dict)

            # Pick difficulty
            difficulty = self._pick_difficulty(rng)

            # Pick template
            template = self._pick_template(rng, task_type, difficulty)
            if template is None:
                # No templates registered for this task type at all -- skip slot
                continue

            # Render question and answer from the template
            try:
                question, answer = template.render(deal_dict)
            except Exception:
                # Template rendering failed (e.g. missing deal fields).
                # Skip this slot rather than emitting a bad record.
                continue

            # Skip empty renders
            if not question or not answer:
                continue

            record = self._build_record(deal_dict, task_type, template, question, answer)
            records.append(record)

        return records
