"""Post-evolution validation: local triple-check + optional Claude gate."""

from __future__ import annotations

import random
from dataclasses import dataclass

from prompt_evolver.fitness import FitnessEvaluator, FitnessResult
from prompt_evolver.genome import Genome, render_markdown
from prompt_evolver.llm import LLMClient


@dataclass
class ValidationResult:
    """Result of a pairwise validation comparison."""

    winner: str  # "evolved" | "original" | "tie"
    votes: list[str]  # individual round votes
    confidence: str  # "unanimous" | "majority" | "split"
    diff_summary: str  # human-readable diff

    @property
    def evolved_wins(self) -> bool:
        return self.winner == "evolved"


_COMPARE_PROMPT = """You are comparing two AI agent prompts. Which would produce more effective, specific, and well-calibrated responses to the scenario below?

SCENARIO:
{scenario}

PROMPT A:
{prompt_a}

PROMPT B:
{prompt_b}

Which prompt is better? Consider:
- Specificity: concrete instructions vs. vague platitudes
- Actionability: clear decision criteria vs. generic advice
- Calibration: appropriate for the context described

Respond with ONLY: A, B, or TIE"""

_COMPARE_PROMPT_ALT = """Evaluate these two agent prompts for quality. Which gives clearer, more actionable guidance for the scenario?

SCENARIO:
{scenario}

FIRST PROMPT:
{prompt_a}

SECOND PROMPT:
{prompt_b}

Which is more effective? Reply with ONLY: A, B, or TIE"""


class Validator:
    """Pairwise validation of evolved prompts against originals."""

    def __init__(self, llm: LLMClient, n_rounds: int = 3):
        self.llm = llm
        self.n_rounds = n_rounds

    def validate(
        self,
        original: Genome,
        evolved: Genome,
        scenarios: list[dict],
        model_override: str | None = None,
    ) -> ValidationResult:
        """Run pairwise validation with position-swapping and majority vote.

        Args:
            original: The production prompt genome.
            evolved: The candidate evolved genome.
            scenarios: Pool of scenarios to test against.
            model_override: Use a different model (e.g., Claude) for validation.

        Returns:
            ValidationResult with winner, votes, confidence, and diff.
        """
        original_md = render_markdown(original)[:3000]
        evolved_md = render_markdown(evolved)[:3000]
        votes: list[str] = []

        for round_idx in range(self.n_rounds):
            scenario = scenarios[round_idx % len(scenarios)] if scenarios else {}
            scenario_text = scenario.get("description", "No scenario provided.")

            # Alternate position and prompt template to reduce bias
            if round_idx % 3 == 0:
                # Original = A, Evolved = B, standard prompt
                prompt = _COMPARE_PROMPT.format(
                    scenario=scenario_text, prompt_a=original_md, prompt_b=evolved_md
                )
                evolved_label = "B"
            elif round_idx % 3 == 1:
                # Evolved = A, Original = B, standard prompt (swapped)
                prompt = _COMPARE_PROMPT.format(
                    scenario=scenario_text, prompt_a=evolved_md, prompt_b=original_md
                )
                evolved_label = "A"
            else:
                # Original = A, Evolved = B, alternate prompt
                prompt = _COMPARE_PROMPT_ALT.format(
                    scenario=scenario_text, prompt_a=original_md, prompt_b=evolved_md
                )
                evolved_label = "B"

            resp = self.llm.complete(
                user=prompt,
                temperature=0.0,
                max_tokens=16,
                model=model_override,
            )

            answer = resp.text.strip().upper()
            if evolved_label in answer:
                votes.append("evolved")
            elif ("A" if evolved_label == "B" else "B") in answer:
                votes.append("original")
            else:
                votes.append("tie")

        # Tally
        evolved_count = votes.count("evolved")
        original_count = votes.count("original")

        if evolved_count > original_count:
            winner = "evolved"
        elif original_count > evolved_count:
            winner = "original"
        else:
            winner = "tie"

        if evolved_count == self.n_rounds or original_count == self.n_rounds:
            confidence = "unanimous"
        elif max(evolved_count, original_count) > self.n_rounds // 2:
            confidence = "majority"
        else:
            confidence = "split"

        diff = generate_diff(original, evolved)

        return ValidationResult(
            winner=winner,
            votes=votes,
            confidence=confidence,
            diff_summary=diff,
        )


def generate_diff(original: Genome, evolved: Genome) -> str:
    """Generate a human-readable segment-by-segment diff."""
    lines = ["# Prompt Diff Report\n"]

    for seg_o in original.segments:
        seg_e = evolved.get_segment(seg_o.key)
        if seg_e is None:
            lines.append(f"## {seg_o.key}: REMOVED in evolved\n")
            continue

        if seg_o.content.strip() == seg_e.content.strip():
            lines.append(f"## {seg_o.key}: unchanged\n")
            continue

        lines.append(f"## {seg_o.key}: CHANGED\n")

        # Show length change
        len_o = len(seg_o.content)
        len_e = len(seg_e.content)
        delta = len_e - len_o
        sign = "+" if delta > 0 else ""
        lines.append(f"Length: {len_o} -> {len_e} ({sign}{delta} chars)\n")

        # Show content
        lines.append(f"### Original:\n{seg_o.content[:500]}\n")
        lines.append(f"### Evolved:\n{seg_e.content[:500]}\n")

    # Check for new segments in evolved
    orig_keys = {s.key for s in original.segments}
    for seg_e in evolved.segments:
        if seg_e.key not in orig_keys:
            lines.append(f"## {seg_e.key}: NEW in evolved\n")
            lines.append(f"{seg_e.content[:500]}\n")

    return "\n".join(lines)
