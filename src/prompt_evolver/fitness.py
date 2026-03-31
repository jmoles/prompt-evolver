"""6-component multi-objective fitness evaluation for agent prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from prompt_evolver.config import FitnessConfig
from prompt_evolver.genome import Genome, render_markdown
from prompt_evolver.llm import LLMClient


@dataclass
class FitnessResult:
    """Scores for all 6 fitness objectives."""

    specificity: float = 0.0
    structure: float = 0.0
    calibration: float = 0.0
    disposition: float = 0.0
    actionability: float = 0.0
    guideline_compliance: float = 0.0

    def as_tuple(self) -> tuple[float, ...]:
        """Return scores as a tuple in objective order."""
        return (
            self.specificity,
            self.structure,
            self.calibration,
            self.disposition,
            self.actionability,
            self.guideline_compliance,
        )

    def as_weighted_array(self, weights) -> np.ndarray:
        """Return negated weighted scores for pymoo minimization."""
        w = (
            weights.specificity,
            weights.structure,
            weights.calibration,
            weights.disposition,
            weights.actionability,
            weights.guideline_compliance,
        )
        scores = self.as_tuple()
        return np.array([-w[i] * scores[i] for i in range(6)])

    def passes_floors(self, floors: dict[str, float]) -> bool:
        """Check if all scores meet minimum thresholds."""
        score_map = {
            "specificity": self.specificity,
            "structure": self.structure,
            "calibration": self.calibration,
            "disposition": self.disposition,
            "actionability": self.actionability,
            "guideline_compliance": self.guideline_compliance,
        }
        for name, floor in floors.items():
            if name in score_map and score_map[name] < floor:
                return False
        return True


class FitnessEvaluator:
    """Evaluates a genome against a scenario using 6 fitness components."""

    def __init__(
        self,
        config: FitnessConfig,
        llm: LLMClient,
        segment_keys: list[str],
        guidelines: list[dict] | None = None,
        reference_genome: Genome | None = None,
    ):
        self.config = config
        self.llm = llm
        self.segment_keys = segment_keys
        self.guidelines = guidelines or []
        self.reference_genome = reference_genome

        # Pre-compile regex patterns
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        for seg_key, patterns in config.specificity_patterns.items():
            self._compiled_patterns[seg_key] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    def evaluate(self, genome: Genome, scenario: dict) -> FitnessResult:
        """Run all 6 fitness components against a genome and scenario.

        Applies length penalty and tone drift penalty as post-processing
        adjustments to the raw scores.

        Args:
            genome: The agent prompt genome to evaluate.
            scenario: A scenario dict with at least 'description' and 'expected_focus'.

        Returns:
            FitnessResult with all 6 scores (0.0 to 1.0 each).
        """
        result = FitnessResult(
            specificity=self._score_specificity(genome, scenario),
            structure=self._score_structure(genome),
            calibration=self._score_calibration(genome),
            disposition=self._score_disposition(genome, scenario),
            actionability=self._score_actionability(genome, scenario),
            guideline_compliance=self._score_guideline_compliance(genome),
        )

        # Apply length penalty — penalize bloated segments
        length_penalty = self._length_penalty(genome)
        result.specificity = max(0.0, result.specificity - length_penalty)
        result.structure = max(0.0, result.structure - length_penalty)

        # Apply tone drift penalty — penalize thesaurus-style rewording
        tone_penalty = self._tone_drift_penalty(genome)
        result.disposition = max(0.0, result.disposition - tone_penalty)
        result.calibration = max(0.0, result.calibration - tone_penalty)

        return result

    def _length_penalty(self, genome: Genome) -> float:
        """Penalize segments that bloat beyond 1.5x their reference length.

        Returns a penalty value (0.0 to 0.3) applied to specificity and structure.
        """
        if not self.reference_genome:
            return 0.0

        penalties = []
        for seg in genome.segments:
            ref_seg = self.reference_genome.get_segment(seg.key)
            if not ref_seg or not ref_seg.content:
                continue
            ratio = len(seg.content) / max(len(ref_seg.content), 1)
            if ratio > 1.5:
                # Linearly penalize from 1.5x (0.0) to 2.5x+ (0.3)
                penalties.append(min((ratio - 1.5) * 0.3, 0.3))

        if not penalties:
            return 0.0
        return max(penalties)  # worst segment drives the penalty

    def _tone_drift_penalty(self, genome: Genome) -> float:
        """Penalize thesaurus-style rewording that inflates without adding value.

        Detects verbose substitutions by checking for filler phrases that
        signal academic paraphrasing rather than direct, opinionated language.
        Returns a penalty (0.0 to 0.2) applied to disposition and calibration.
        """
        verbose_markers = [
            "in this scenario",
            "it is crucial to",
            "it's crucial to",
            "it is important to",
            "it's important to",
            "delving into",
            "exploring strategies",
            "investigating the",
            "commence",
            "orchestrating",
            "embody",
            "concurrently",
            "respectively",
            "elaborate response",
            "the crux of the matter",
            "foster",
            "emphasizing their ability",
            "potential impact on",
            "ensuring a seamless",
            "in the form of",
        ]

        full_text = render_markdown(genome).lower()
        hits = sum(1 for m in verbose_markers if m in full_text)
        return min(hits * 0.05, 0.2)

    def _score_specificity(self, genome: Genome, scenario: dict) -> float:
        """Deterministic: regex pattern matching + anti-vagueness penalty."""
        total_score = 0.0
        total_checks = 0

        # Check specificity patterns per segment
        for seg_key, patterns in self._compiled_patterns.items():
            seg = genome.get_segment(seg_key)
            if seg is None or not patterns:
                continue
            hits = sum(1 for p in patterns if p.search(seg.content))
            total_score += hits / len(patterns)
            total_checks += 1

        # Check expected entities from scenario
        expected = scenario.get("expected_focus", [])
        if expected:
            full_text = render_markdown(genome).lower()
            found = sum(1 for e in expected if e.lower() in full_text)
            total_score += found / len(expected)
            total_checks += 1

        # Vagueness penalty
        full_text = render_markdown(genome).lower()
        vague_count = sum(
            1 for vp in self.config.vague_phrases if vp.lower() in full_text
        )
        vagueness_penalty = min(vague_count * 0.1, 0.5)

        if total_checks > 0:
            return max(0.0, (total_score / total_checks) - vagueness_penalty)
        return 0.0

    def _score_structure(self, genome: Genome) -> float:
        """Deterministic: required segments present and properly formatted."""
        present_keys = set(genome.segment_keys())
        required_keys = [k for k in self.segment_keys if k != "role_statement"]
        # For simplicity, treat all configured keys as expected
        if not required_keys:
            return 1.0

        found = sum(1 for k in required_keys if k in present_keys)
        base_score = found / len(required_keys)

        # Bonus: check that response_style segment contains decision rules
        style = genome.get_segment("response_style")
        if style and any(
            w in style.content.lower() for w in ["lead with", "never", "always", "first"]
        ):
            base_score = min(1.0, base_score + 0.1)

        # Penalty: filler openings in role_statement
        role = genome.get_segment("role_statement")
        if role:
            filler_starts = [
                "great question",
                "sure,",
                "certainly",
                "thank you for",
                "i'd be happy to",
            ]
            if any(role.content.lower().strip().startswith(f) for f in filler_starts):
                base_score = max(0.0, base_score - 0.2)

        return min(1.0, base_score)

    def _score_calibration(self, genome: Genome) -> float:
        """Semi-deterministic: keyword density for domain calibration."""
        total_score = 0.0
        total_checks = 0

        for seg_key, keywords in self.config.calibration_keywords.items():
            seg = genome.get_segment(seg_key)
            if seg is None or not keywords:
                continue
            content_lower = seg.content.lower()
            hits = sum(1 for kw in keywords if kw.lower() in content_lower)
            total_score += min(hits / max(len(keywords), 1), 1.0)
            total_checks += 1

        if total_checks > 0:
            return total_score / total_checks
        return 0.5  # neutral when no calibration keywords configured

    def _score_disposition(self, genome: Genome, scenario: dict) -> float:
        """LLM-judged: does the disposition match the scenario needs?"""
        rubric = self.config.llm_prompts.get("disposition_rubric", "")
        if not rubric:
            return 0.5  # neutral when no rubric configured

        disposition_seg = genome.get_segment("disposition")
        disposition_text = disposition_seg.content if disposition_seg else ""
        scenario_text = scenario.get("description", "")

        prompt = rubric.format(
            disposition=disposition_text[:1500],
            scenario=scenario_text[:500],
        )

        score = self.llm.judge_score(prompt)
        return (score - 1) / 4.0  # normalize 1-5 to 0.0-1.0

    def _score_actionability(self, genome: Genome, scenario: dict) -> float:
        """LLM-judged: how many concrete, actionable instructions?"""
        rubric = self.config.llm_prompts.get("actionability_rubric", "")
        if not rubric:
            return 0.5

        full_prompt = render_markdown(genome)[:2000]
        scenario_text = scenario.get("description", "")

        prompt = rubric.format(
            prompt=full_prompt,
            scenario=scenario_text[:500],
        )

        resp = self.llm.complete(user=prompt, temperature=0.0, max_tokens=16)
        try:
            match = re.search(r"\d+", resp.text)
            if match:
                count = int(match.group())
                return min(count * 0.15, 1.0)  # diminishing returns
        except (ValueError, IndexError):
            pass
        return 0.3  # default

    def _score_guideline_compliance(self, genome: Genome) -> float:
        """Semi-deterministic: does the prompt incorporate known guidelines?"""
        if not self.guidelines:
            # Fall back to config-based guideline keywords
            return self._score_guideline_keywords(genome)

        full_text = render_markdown(genome).lower()
        score = 0.0
        applicable = 0

        for g in self.guidelines:
            keywords = g.get("keywords", [])
            polarity = g.get("polarity", "positive")
            if not keywords:
                continue

            present = any(kw.lower() in full_text for kw in keywords)
            if polarity == "positive":
                score += 1.0 if present else 0.0
            else:
                score += 0.0 if present else 1.0
            applicable += 1

        if applicable > 0:
            return score / applicable
        return 0.5

    def _score_guideline_keywords(self, genome: Genome) -> float:
        """Fallback guideline scoring using config keywords."""
        total_score = 0.0
        total_checks = 0

        for seg_key, keywords in self.config.guideline_keywords.items():
            seg = genome.get_segment(seg_key)
            if seg is None or not keywords:
                continue
            content_lower = seg.content.lower()
            hits = sum(1 for kw in keywords if kw.lower() in content_lower)
            total_score += min(hits / max(len(keywords), 1), 1.0)
            total_checks += 1

        if total_checks > 0:
            return total_score / total_checks
        return 0.5
