from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from config import MODEL_ANALYSIS, MODEL_DECISION, OPENAI_API_KEY

LOGGER = logging.getLogger(__name__)


def _parse_json_object(raw_text: str) -> dict[str, Any] | None:
    """Parse a JSON object, tolerating markdown fences and surrounding prose.

    Models occasionally wrap output in ```json fences even when a JSON
    response format is requested; without this tolerance the whole analysis
    silently degrades to its fallback.
    """
    text = raw_text.strip()
    if not text:
        return None

    candidates: list[str] = [text]

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for chunk in fenced:
        stripped = chunk.strip()
        if stripped:
            candidates.append(stripped)

    first_obj = text.find("{")
    last_obj = text.rfind("}")
    if first_obj != -1 and first_obj < last_obj:
        candidates.append(text[first_obj : last_obj + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None

try:
    from openai import OpenAI  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - environment dependent
    OpenAI = None


@dataclass(frozen=True)
class LLMUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMResult:
    ok: bool
    payload: dict[str, Any]
    usage: LLMUsage
    model: str
    error: str


class LLMClient:
    def __init__(self, api_key: str = OPENAI_API_KEY) -> None:
        self._api_key = api_key
        self._client = OpenAI(api_key=api_key) if (OpenAI is not None and api_key) else None

    def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        fallback_payload: dict[str, Any],
    ) -> LLMResult:
        if self._client is None:
            return LLMResult(
                ok=False,
                payload=fallback_payload,
                usage=LLMUsage(0, 0, 0),
                model=model,
                error="OpenAI client unavailable",
            )

        try:
            response = self._client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text={"format": {"type": "json_object"}},
            )

            output_text = getattr(response, "output_text", "") or ""
            usage_obj = getattr(response, "usage", None)
            usage = LLMUsage(
                prompt_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
                completion_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
                total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
            )

            if not output_text.strip():
                return LLMResult(
                    ok=False,
                    payload=fallback_payload,
                    usage=usage,
                    model=model,
                    error="empty output",
                )

            payload = _parse_json_object(output_text)
            if payload is None:
                LOGGER.warning("LLM JSON parse failed: head=%s", output_text[:200])
                return LLMResult(
                    ok=False,
                    payload=fallback_payload,
                    usage=usage,
                    model=model,
                    error="json parse error",
                )

            return LLMResult(
                ok=True,
                payload=payload,
                usage=usage,
                model=model,
                error="",
            )
        except Exception as exc:
            LOGGER.warning("LLM call failed: %s", exc)
            return LLMResult(
                ok=False,
                payload=fallback_payload,
                usage=LLMUsage(0, 0, 0),
                model=model,
                error=str(exc),
            )

    def call_function(
        self,
        system_prompt: str,
        user_prompt: str,
        model: str,
        function_name: str,
        function_schema: dict[str, Any],
        fallback_payload: dict[str, Any],
    ) -> LLMResult:
        if self._client is None:
            return LLMResult(
                ok=False,
                payload=fallback_payload,
                usage=LLMUsage(0, 0, 0),
                model=model,
                error="OpenAI client unavailable",
            )

        try:
            response = self._client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=[
                    {
                        "type": "function",
                        "name": function_name,
                        "description": str(function_schema.get("description", "")),
                        "parameters": function_schema.get("parameters", {}),
                    }
                ],
                tool_choice={"type": "function", "name": function_name},
            )

            payload = fallback_payload
            found_call = False
            output_items = getattr(response, "output", []) or []
            for item in output_items:
                item_type = getattr(item, "type", "")
                item_name = getattr(item, "name", "")
                if item_type == "function_call" and item_name == function_name:
                    found_call = True
                    arguments = str(getattr(item, "arguments", "") or "")
                    if arguments:
                        try:
                            payload = json.loads(arguments)
                        except json.JSONDecodeError as exc:
                            LOGGER.warning("LLM function arguments parse failed: %s", exc)
                    break

            usage_obj = getattr(response, "usage", None)
            usage = LLMUsage(
                prompt_tokens=int(getattr(usage_obj, "input_tokens", 0) or 0),
                completion_tokens=int(getattr(usage_obj, "output_tokens", 0) or 0),
                total_tokens=int(getattr(usage_obj, "total_tokens", 0) or 0),
            )

            if not found_call:
                return LLMResult(
                    ok=False,
                    payload=fallback_payload,
                    usage=usage,
                    model=model,
                    error="function_call not found",
                )

            if payload is fallback_payload:
                return LLMResult(
                    ok=False,
                    payload=fallback_payload,
                    usage=usage,
                    model=model,
                    error="json parse error",
                )

            return LLMResult(
                ok=found_call,
                payload=payload,
                usage=usage,
                model=model,
                error="",
            )
        except Exception as exc:
            LOGGER.warning("LLM function call failed: %s", exc)
            return LLMResult(
                ok=False,
                payload=fallback_payload,
                usage=LLMUsage(0, 0, 0),
                model=model,
                error=str(exc),
            )


def get_default_client() -> LLMClient:
    return LLMClient()


def analysis_model() -> str:
    return MODEL_ANALYSIS


def decision_model() -> str:
    return MODEL_DECISION
