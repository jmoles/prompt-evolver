"""Thin LLM client wrapping LiteLLM for local (Ollama) and cloud (Claude) models."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import litellm


# Suppress litellm's verbose logging by default
litellm.suppress_debug_info = True


@dataclass
class LLMResponse:
    """Response from an LLM call."""

    text: str
    model: str
    usage: dict


class LLMClient:
    """Unified LLM client for mutation, evaluation, and validation."""

    def __init__(
        self,
        model: str = "ollama/mistral:7b-instruct-v0.3-q5_K_M",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        timeout: int = 120,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def complete(
        self,
        user: str,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResponse:
        """Send a completion request.

        Args:
            user: The user message / prompt.
            system: Optional system message.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.
            model: Override default model for this call.

        Returns:
            LLMResponse with text, model used, and usage stats.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        response = litellm.completion(
            model=model or self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            timeout=self.timeout,
        )

        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            }

        return LLMResponse(
            text=choice.message.content or "",
            model=response.model or self.model,
            usage=usage,
        )

    def complete_json(
        self,
        user: str,
        system: str | None = None,
        model: str | None = None,
    ) -> dict:
        """Send a completion request expecting JSON output.

        Attempts to parse JSON from the response. Falls back to extracting
        JSON blocks from freeform text for models that don't support
        structured output natively.

        Args:
            user: The user message (should request JSON output).
            system: Optional system message.
            model: Override default model.

        Returns:
            Parsed JSON as a dict.

        Raises:
            ValueError: If no valid JSON could be extracted.
        """
        resp = self.complete(
            user=user,
            system=system,
            model=model,
            temperature=0.0,
        )
        return _extract_json(resp.text)

    def judge_score(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 16,
    ) -> int:
        """Send a judging prompt expecting a single integer score (1-5).

        Args:
            prompt: The full judging prompt.
            temperature: Defaults to 0.0 for deterministic judging.
            max_tokens: Small — we only need a number.

        Returns:
            Integer score, or 3 (neutral) on parse failure.
        """
        resp = self.complete(
            user=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            # Extract first digit from response
            match = re.search(r"[1-5]", resp.text)
            if match:
                return int(match.group())
        except (ValueError, IndexError):
            pass
        return 3  # neutral default


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM text output.

    Tries direct parse first, then looks for ```json blocks,
    then searches for {} or [] patterns.
    """
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Look for ```json ... ``` blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Look for first { ... } or [ ... ]
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:200]}")
