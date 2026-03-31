"""Post-evolution analysis: convergence tracking, diff reports, guideline extraction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from prompt_evolver.fitness import FitnessResult
from prompt_evolver.genome import Genome, render_markdown
from prompt_evolver.llm import LLMClient
from prompt_evolver.validator import generate_diff


class EvolutionAnalyzer:
    """Track and analyze evolution run results."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.convergence_log: list[dict] = []

    def track_generation(
        self,
        gen: int,
        fitness_results: list[FitnessResult],
    ) -> None:
        """Log stats for a generation."""
        if not fitness_results:
            return

        n = len(fitness_results)
        obj_names = [
            "specificity", "structure", "calibration",
            "disposition", "actionability", "guideline_compliance",
        ]

        entry = {
            "generation": gen,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pop_size": n,
        }

        for i, name in enumerate(obj_names):
            values = [r.as_tuple()[i] for r in fitness_results]
            entry[f"{name}_mean"] = sum(values) / n
            entry[f"{name}_max"] = max(values)
            entry[f"{name}_min"] = min(values)

        self.convergence_log.append(entry)

    def save_convergence_log(self) -> None:
        """Write convergence log to disk."""
        path = self.output_dir / "convergence.json"
        with open(path, "w") as f:
            json.dump(self.convergence_log, f, indent=2)

    def convergence_report(self) -> str:
        """Generate a human-readable convergence summary."""
        if not self.convergence_log:
            return "No convergence data recorded."

        lines = ["# Convergence Report\n"]
        first = self.convergence_log[0]
        last = self.convergence_log[-1]

        lines.append(f"Generations: {first['generation']} to {last['generation']}")
        lines.append(f"Population: {last['pop_size']}\n")

        obj_names = [
            "specificity", "structure", "calibration",
            "disposition", "actionability", "guideline_compliance",
        ]

        lines.append("| Objective | Start Mean | End Mean | Delta |")
        lines.append("|-----------|-----------|---------|-------|")

        for name in obj_names:
            start = first.get(f"{name}_mean", 0)
            end = last.get(f"{name}_mean", 0)
            delta = end - start
            sign = "+" if delta >= 0 else ""
            lines.append(f"| {name} | {start:.3f} | {end:.3f} | {sign}{delta:.3f} |")

        return "\n".join(lines)

    def segment_diff_report(self, original: Genome, evolved: Genome) -> str:
        """Generate segment-by-segment diff between original and evolved."""
        return generate_diff(original, evolved)

    def save_genome(self, genome: Genome, filename: str) -> None:
        """Save a genome as a markdown file."""
        path = self.output_dir / filename
        with open(path, "w") as f:
            f.write(render_markdown(genome))

    def extract_guidelines(
        self,
        original: Genome,
        evolved: Genome,
        llm: LLMClient,
    ) -> list[dict]:
        """Compare original and evolved genomes, extract improvement guidelines.

        Uses the LLM to articulate what changed and why it might be better.
        Returns candidate guidelines for the guidelines DB.
        """
        original_md = render_markdown(original)[:2000]
        evolved_md = render_markdown(evolved)[:2000]

        prompt = (
            "Compare these two versions of an agent prompt. The EVOLVED version "
            "scored higher in fitness evaluation.\n\n"
            f"ORIGINAL:\n{original_md}\n\n"
            f"EVOLVED:\n{evolved_md}\n\n"
            "Extract 3-5 specific, actionable guidelines that explain WHY the "
            "evolved version is better. Each guideline should be:\n"
            "- A concrete instruction (not 'be more specific' but a specific rule)\n"
            "- Testable (you could check whether a prompt follows it)\n"
            "- Segment-specific (state which prompt section it applies to)\n\n"
            'Output as JSON array: [{"text": "...", "segment_key": "...", '
            '"keywords": ["...", "..."], "polarity": "positive"}]'
        )

        try:
            data = llm.complete_json(user=prompt)
            items = data if isinstance(data, list) else []
            return items
        except (ValueError, KeyError):
            return []

    def pareto_front_summary(
        self,
        genomes: list[Genome],
        fitness_results: list[FitnessResult],
    ) -> str:
        """Summarize the Pareto-optimal solutions."""
        lines = [f"# Pareto Front Summary\n\nSolutions: {len(genomes)}\n"]

        for i, (genome, fit) in enumerate(zip(genomes, fitness_results)):
            scores = fit.as_tuple()
            obj_names = [
                "specificity", "structure", "calibration",
                "disposition", "actionability", "guideline_compliance",
            ]
            score_str = ", ".join(f"{n}={s:.2f}" for n, s in zip(obj_names, scores))
            lines.append(f"## Solution {i + 1}\n{score_str}\n")

        return "\n".join(lines)
