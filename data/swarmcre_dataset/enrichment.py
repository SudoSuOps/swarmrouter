"""
SwarmCRE Dataset Factory — Optional Together.ai Enrichment

Decorator pattern: takes a record in, returns a record out (potentially
with an enriched assistant answer via LLM rewrite + CoVe verification).

Enrichment targets:
    - ic_memo           : rewrite for professional IC-quality prose
    - lease_reasoning   : add nuanced negotiation strategy detail
    - market_comp_narrative : deepen submarket analysis and comp adjustments

Rate limiting, exponential backoff, and think-block stripping are handled
internally.  If the enrichment service is unavailable or disabled, records
pass through unchanged.
"""

import json
import logging
import os
import re
import time
from typing import Optional

try:
    import urllib.request
    import urllib.error
    _HAS_URLLIB = True
except ImportError:
    _HAS_URLLIB = False

log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# MODEL CONSTANTS
# ═══════════════════════════════════════════════════════════════════

GEN_MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
VERIFY_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507-tput"
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"

# Rate limit: minimum 0.08s between API calls
_MIN_CALL_INTERVAL = 0.08

# Timeout for each API call
_API_TIMEOUT = 180

# Maximum retries on 429
_MAX_RETRIES = 5

# Task types eligible for enrichment
_ENRICHABLE_TASKS = frozenset({"ic_memo", "lease_reasoning", "market_comp_narrative"})

# Regex to strip Qwen3 think blocks: <think>...</think>
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# ═══════════════════════════════════════════════════════════════════
# TOGETHER CLIENT
# ═══════════════════════════════════════════════════════════════════


class _TogetherClient:
    """Minimal Together.ai API client with rate limiting and backoff.

    Reuses the pattern from cre_mega_grinder.py:
        - 0.08s minimum between calls
        - Exponential backoff on HTTP 429
        - 180s timeout
        - Think-block stripping for Qwen3 models
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._last_call_time: float = 0.0

    def _rate_limit(self):
        """Enforce minimum interval between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < _MIN_CALL_INTERVAL:
            time.sleep(_MIN_CALL_INTERVAL - elapsed)
        self._last_call_time = time.time()

    def _strip_think_blocks(self, text: str) -> str:
        """Remove <think>...</think> blocks from Qwen3 output."""
        return _THINK_BLOCK_RE.sub("", text).strip()

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request to Together.ai.

        Returns the assistant content string. Raises on unrecoverable errors.
        """
        if not _HAS_URLLIB:
            raise RuntimeError("urllib not available for Together.ai API calls")

        payload = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        last_error = None
        for attempt in range(_MAX_RETRIES):
            self._rate_limit()

            req = urllib.request.Request(
                TOGETHER_URL,
                data=payload,
                headers=headers,
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=_API_TIMEOUT) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    content = body["choices"][0]["message"]["content"]

                    # Strip think blocks for Qwen3 models
                    if "qwen" in model.lower() or "Qwen" in model:
                        content = self._strip_think_blocks(content)

                    return content

            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 429:
                    # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                    backoff = 2 ** attempt
                    log.warning(
                        "Together.ai 429 rate limit, attempt %d/%d, "
                        "backing off %ds",
                        attempt + 1, _MAX_RETRIES, backoff,
                    )
                    time.sleep(backoff)
                    continue
                else:
                    log.error("Together.ai HTTP %d: %s", e.code, e.reason)
                    raise

            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_error = e
                backoff = 2 ** attempt
                log.warning(
                    "Together.ai network error on attempt %d/%d: %s, "
                    "backing off %ds",
                    attempt + 1, _MAX_RETRIES, e, backoff,
                )
                time.sleep(backoff)
                continue

        raise RuntimeError(
            f"Together.ai request failed after {_MAX_RETRIES} retries: "
            f"{last_error}"
        )


# ═══════════════════════════════════════════════════════════════════
# ENRICHMENT PROMPTS
# ═══════════════════════════════════════════════════════════════════

_REWRITE_PROMPTS = {
    "ic_memo": (
        "You are a senior CRE investment analyst at a top-tier institutional fund. "
        "Rewrite the following Investment Committee memo to be more precise, "
        "decision-ready, and institutional-quality. Preserve all financial figures "
        "exactly. Improve prose quality, structure, and analytical depth. "
        "Keep the same recommendation.\n\n"
        "ORIGINAL MEMO:\n{answer}\n\n"
        "Rewrite the memo. Output ONLY the improved memo text."
    ),
    "lease_reasoning": (
        "You are a 30-year CRE lease negotiation expert. Enhance the following "
        "lease analysis with deeper market context, tenant leverage assessment, "
        "and specific negotiation tactics. Preserve all dollar amounts and "
        "percentages exactly.\n\n"
        "ORIGINAL ANALYSIS:\n{answer}\n\n"
        "Provide the enhanced analysis. Output ONLY the improved text."
    ),
    "market_comp_narrative": (
        "You are a senior CRE appraiser and market analyst. Enhance the following "
        "comparable transaction analysis with deeper submarket narrative, "
        "adjustment rationale, and market trend context. Preserve all numbers "
        "and comp data exactly.\n\n"
        "ORIGINAL ANALYSIS:\n{answer}\n\n"
        "Provide the enhanced analysis. Output ONLY the improved text."
    ),
}

_COVE_PROMPT = (
    "You are a fact-checking AI. Compare the ORIGINAL and REWRITTEN versions below. "
    "Verify that:\n"
    "1. All numeric values (dollar amounts, percentages, ratios) are preserved exactly\n"
    "2. No factual claims are added that contradict the original\n"
    "3. The recommendation/conclusion is the same\n"
    "4. The rewrite is higher quality than the original\n\n"
    "ORIGINAL:\n{original}\n\n"
    "REWRITTEN:\n{rewritten}\n\n"
    "Respond with a JSON object:\n"
    '{{"numbers_preserved": true/false, "facts_consistent": true/false, '
    '"recommendation_same": true/false, "quality_improved": true/false, '
    '"overall_pass": true/false, "issues": ["list of issues if any"]}}'
)


# ═══════════════════════════════════════════════════════════════════
# ENRICHER
# ═══════════════════════════════════════════════════════════════════


class Enricher:
    """Optional Together.ai enrichment for high-value narrative tasks.

    Decorator pattern: takes a record dict in, returns a record dict out.
    If enrichment is disabled, the API key is missing, or the task type
    is not enrichable, the record passes through unchanged.

    Usage::

        enricher = Enricher(enabled=True, api_key="...")
        enriched_record = enricher.enrich(record)
    """

    def __init__(
        self,
        enabled: bool = True,
        api_key: str = "",
        log: Optional[logging.Logger] = None,
    ):
        self.enabled = enabled
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY", "")
        self.log = log or logging.getLogger(__name__)
        self._client: Optional[_TogetherClient] = None

        # Stats tracking
        self.stats = {
            "attempted": 0,
            "enriched": 0,
            "cove_passed": 0,
            "cove_failed": 0,
            "errors": 0,
            "skipped": 0,
        }

        if self.enabled and self.api_key:
            self._client = _TogetherClient(self.api_key)
            self.log.info("Enricher initialized with Together.ai client")
        else:
            self.log.info(
                "Enricher disabled (enabled=%s, has_key=%s)",
                self.enabled, bool(self.api_key),
            )

    def _is_enrichable(self, record: dict) -> bool:
        """Check if a record is eligible for enrichment."""
        task_type = record.get("task_type", "")
        return task_type in _ENRICHABLE_TASKS

    def _extract_answer(self, record: dict) -> str:
        """Get assistant answer from messages."""
        for msg in reversed(record.get("messages", [])):
            if msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""

    def _set_answer(self, record: dict, new_answer: str) -> dict:
        """Replace assistant answer in the record (returns new dict)."""
        record = dict(record)
        messages = list(record.get("messages", []))
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                messages[i] = dict(messages[i])
                messages[i]["content"] = new_answer
                break
        record["messages"] = messages
        return record

    def _rewrite(self, task_type: str, answer: str) -> str:
        """Use 70B-Turbo to rewrite the answer for higher quality."""
        prompt_template = _REWRITE_PROMPTS.get(task_type)
        if not prompt_template:
            return answer

        prompt = prompt_template.format(answer=answer)
        messages = [
            {"role": "system", "content": "You are an expert CRE analyst."},
            {"role": "user", "content": prompt},
        ]

        return self._client.chat(
            model=GEN_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
        )

    def _verify_cove(self, original: str, rewritten: str) -> dict:
        """Use 235B for Chain-of-Verification (CoVe) on the rewrite.

        Returns a dict with verification scores.
        """
        prompt = _COVE_PROMPT.format(original=original, rewritten=rewritten)
        messages = [
            {
                "role": "system",
                "content": "You are a precise fact-checking AI. Respond only with valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        response = self._client.chat(
            model=VERIFY_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )

        # Parse JSON response
        try:
            # Handle potential markdown code fences
            cleaned = response.strip()
            if cleaned.startswith("```"):
                # Remove code fences
                lines = cleaned.split("\n")
                cleaned = "\n".join(
                    l for l in lines
                    if not l.strip().startswith("```")
                )
            result = json.loads(cleaned)
            return result
        except (json.JSONDecodeError, KeyError):
            self.log.warning("CoVe response was not valid JSON: %s", response[:200])
            return {
                "numbers_preserved": False,
                "facts_consistent": False,
                "recommendation_same": False,
                "quality_improved": False,
                "overall_pass": False,
                "issues": ["CoVe response parsing failed"],
            }

    def enrich(self, record: dict) -> dict:
        """Enrich a record via Together.ai if eligible.

        If the task type is not enrichable, enrichment is disabled,
        or no API key is available, the record passes through unchanged.

        The enrichment pipeline:
            1. Rewrite the answer using 70B-Turbo for higher quality
            2. Verify the rewrite using 235B CoVe (Chain of Verification)
            3. If CoVe passes, use the rewritten answer
            4. If CoVe fails, keep the original answer

        Metadata is added: enriched=True/False, cove_score={...}
        """
        # Pass-through conditions
        if not self.enabled or not self._client:
            self.stats["skipped"] += 1
            return record

        if not self._is_enrichable(record):
            self.stats["skipped"] += 1
            return record

        task_type = record.get("task_type", "")
        original_answer = self._extract_answer(record)

        if not original_answer or len(original_answer) < 50:
            self.stats["skipped"] += 1
            return record

        self.stats["attempted"] += 1

        try:
            # Step 1: Rewrite with 70B-Turbo
            rewritten = self._rewrite(task_type, original_answer)

            if not rewritten or len(rewritten) < 50:
                self.log.warning("Rewrite produced empty/short output for %s", record.get("id", ""))
                self.stats["errors"] += 1
                return record

            # Step 2: CoVe verification with 235B
            cove_result = self._verify_cove(original_answer, rewritten)
            cove_passed = cove_result.get("overall_pass", False)

            if cove_passed:
                # Use the enriched answer
                record = self._set_answer(record, rewritten)
                record["enriched"] = True
                record["cove_score"] = cove_result
                self.stats["enriched"] += 1
                self.stats["cove_passed"] += 1
                self.log.debug(
                    "Enriched record %s (task=%s)",
                    record.get("id", ""), task_type,
                )
            else:
                # Keep original, but note the attempt
                record = dict(record)
                record["enriched"] = False
                record["cove_score"] = cove_result
                self.stats["cove_failed"] += 1
                self.log.debug(
                    "CoVe rejected rewrite for %s: %s",
                    record.get("id", ""),
                    cove_result.get("issues", []),
                )

            return record

        except Exception as e:
            self.log.error(
                "Enrichment failed for %s: %s",
                record.get("id", ""), e,
            )
            self.stats["errors"] += 1
            # Return original record unchanged on error
            return record
