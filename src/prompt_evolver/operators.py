"""Evolutionary operators: mutation, crossover, sampling for pymoo."""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

import numpy as np
from pymoo.core.crossover import Crossover
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling

from prompt_evolver.genome import Genome
from prompt_evolver.llm import LLMClient


# Patterns that indicate the LLM echoed back mutation prompt framing
_PROMPT_LEAK_PATTERNS = [
    re.compile(r"(?i)^(?:MODIFIED|UPDATED|REWRITTEN|PRUNED)\s*VERSION\s*[:\-]?\s*", re.MULTILINE),
    re.compile(r"(?i)(?:In this case|Here's a modified|The changes made|transformed like A was transformed into B).*?\n", re.MULTILINE),
    re.compile(r"(?i)^(?:VERSION [ABC]|CURRENT VERSION|ORIGINAL)[:\-]\s*$", re.MULTILINE),
    re.compile(r"(?i)^(?:GUIDELINE TO ADD|CURRENT TEXT|UPDATED TEXT)[:\-]\s*$", re.MULTILINE),
]


def _sanitize_mutation_output(text: str) -> str:
    """Remove LLM prompt artifacts that leak into mutation output."""
    text = text.strip()
    for pattern in _PROMPT_LEAK_PATTERNS:
        text = pattern.sub("", text)
    # Remove leading/trailing whitespace left by removals
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

if TYPE_CHECKING:
    from prompt_evolver.guidelines import GuidelinesDB


class PromptSampling(Sampling):
    """Initialize population from a seed genome with light mutations."""

    def __init__(self, seed_genome: Genome, llm: LLMClient):
        super().__init__()
        self.seed_genome = seed_genome
        self.llm = llm

    def _do(self, problem, n_samples, **kwargs):
        X = np.empty((n_samples, 1), dtype=object)

        # First individual is the unmodified seed
        X[0, 0] = self.seed_genome.copy()

        # Remaining individuals are light mutations of the seed
        for i in range(1, n_samples):
            genome = self.seed_genome.copy()
            # Pick a random segment and paraphrase it
            segments = [s for s in genome.segments if s.key != "role_statement"]
            if segments:
                seg = random.choice(segments)
                prompt = (
                    "Rewrite the following text to say the same thing differently. "
                    "Keep the same meaning, specificity, and technical depth. "
                    "Change sentence structure and word choice.\n\n"
                    f"ORIGINAL:\n{seg.content}\n\nREWRITTEN:"
                )
                resp = self.llm.complete(user=prompt, temperature=0.8, max_tokens=512)
                text = resp.text.strip()
                if len(text) > 50:  # sanity check
                    genome.set_segment(seg.key, text)
            X[i, 0] = genome

        return X


class PromptMutation(Mutation):
    """Three-strategy mutation: DE difference, guideline injection, pruning."""

    def __init__(
        self,
        llm: LLMClient,
        guidelines_db: GuidelinesDB | None = None,
        de_diff_weight: float = 0.60,
        guideline_weight: float = 0.25,
        prune_weight: float = 0.15,
    ):
        super().__init__()
        self.llm = llm
        self.guidelines_db = guidelines_db
        self.de_diff_weight = de_diff_weight
        self.guideline_weight = guideline_weight
        self.prune_weight = prune_weight

    def _do(self, problem, X, **kwargs):
        Xp = np.copy(X)
        population = [row[0] for row in X]

        for i in range(len(Xp)):
            genome: Genome = Xp[i, 0].copy()

            # Select mutation strategy by weight
            strategy = random.choices(
                ["de_diff", "guideline", "prune"],
                weights=[self.de_diff_weight, self.guideline_weight, self.prune_weight],
            )[0]

            if strategy == "de_diff" and len(population) >= 3:
                genome = self._de_diff_mutate(genome, population)
            elif strategy == "guideline" and self.guidelines_db:
                genome = self._guideline_inject(genome)
            else:
                genome = self._prune(genome)

            Xp[i, 0] = genome

        return Xp

    def _de_diff_mutate(self, target: Genome, population: list[Genome]) -> Genome:
        """DE-inspired difference mutation: learn transformation from two donors."""
        # Pick two random donors different from target
        donors = random.sample(population, min(2, len(population)))
        if len(donors) < 2:
            return self._prune(target)

        donor_a, donor_b = donors

        # Pick a random evolvable segment
        segments = [s for s in target.segments if s.key != "role_statement"]
        if not segments:
            return target

        seg = random.choice(segments)
        seg_a = donor_a.get_segment(seg.key)
        seg_b = donor_b.get_segment(seg.key)

        if not seg_a or not seg_b:
            return target

        prompt = (
            "Below are three versions of the same section of an agent prompt.\n\n"
            f"VERSION A:\n{seg_a.content[:600]}\n\n"
            f"VERSION B:\n{seg_b.content[:600]}\n\n"
            f"CURRENT VERSION:\n{seg.content[:600]}\n\n"
            "Identify what changed between Version A and Version B "
            "(tone, specificity, rules, structure). Apply a SIMILAR type of "
            "change to the Current Version. Keep the same domain content but "
            "transform it the way A was transformed into B.\n\n"
            "MODIFIED VERSION:"
        )

        resp = self.llm.complete(user=prompt, temperature=0.7, max_tokens=512)
        text = _sanitize_mutation_output(resp.text)
        if len(text) > 50:
            target.set_segment(seg.key, text)

        return target

    def _guideline_inject(self, target: Genome) -> Genome:
        """Inject a learned guideline into a relevant segment."""
        if not self.guidelines_db:
            return self._prune(target)

        guidelines = self.guidelines_db.get_random(n=1)
        if not guidelines:
            return self._prune(target)

        guideline = guidelines[0]
        seg_key = guideline.get("segment_key", "")
        seg = target.get_segment(seg_key) if seg_key else None

        # Fall back to a random segment if the guideline's target isn't found
        if seg is None:
            segments = [s for s in target.segments if s.key != "role_statement"]
            if not segments:
                return target
            seg = random.choice(segments)

        prompt = (
            "Add the following guideline to this agent prompt section. "
            "Integrate it naturally — do not just append it. "
            "Adjust existing text if needed to avoid contradiction.\n\n"
            f"GUIDELINE TO ADD:\n{guideline.get('text', '')}\n\n"
            f"CURRENT TEXT:\n{seg.content[:600]}\n\n"
            "UPDATED TEXT:"
        )

        resp = self.llm.complete(user=prompt, temperature=0.5, max_tokens=512)
        text = _sanitize_mutation_output(resp.text)
        if len(text) > 50:
            target.set_segment(seg.key, text)

        return target

    def _prune(self, target: Genome) -> Genome:
        """Remove the weakest/most generic instruction from the longest segment."""
        segments = [s for s in target.segments if s.key != "role_statement"]
        if not segments:
            return target

        # Pick the longest segment
        seg = max(segments, key=lambda s: len(s.content))
        if len(seg.content) < 200:
            return target  # too short to prune

        prompt = (
            "This agent prompt section has grown too long. "
            "Remove ONE instruction, rule, or sentence that is:\n"
            "- The most generic (could apply to any agent, not this one specifically)\n"
            "- OR redundant with another instruction already present\n"
            "- OR the weakest/vaguest statement\n\n"
            "Keep everything else exactly as-is.\n\n"
            f"CURRENT TEXT:\n{seg.content}\n\n"
            "PRUNED TEXT:"
        )

        resp = self.llm.complete(user=prompt, temperature=0.3, max_tokens=512)
        text = _sanitize_mutation_output(resp.text)
        if 50 < len(text) < len(seg.content):
            target.set_segment(seg.key, text)

        return target


class SegmentCrossover(Crossover):
    """Uniform crossover at the segment level."""

    def __init__(self, swap_prob: float = 0.3):
        # 2 parents in, 2 offspring out
        super().__init__(n_parents=2, n_offsprings=2)
        self.swap_prob = swap_prob

    def _extract_genome(self, x) -> Genome:
        """Safely extract a Genome from whatever pymoo gives us."""
        if isinstance(x, Genome):
            return x.copy()
        if isinstance(x, np.ndarray):
            # Could be array([Genome]) or nested
            flat = x.flat
            for item in flat:
                if isinstance(item, Genome):
                    return item.copy()
        return x.copy()

    def _do(self, problem, X, **kwargs):
        # pymoo crossover passes X as (n_matings, n_parents, n_var).
        # With object dtype and n_var=1, shapes can be inconsistent.
        # Use the parent Population objects passed via kwargs if available,
        # otherwise safely navigate the array.

        # pymoo swapaxes the input so X is (n_parents, n_matings, n_var).
        # Output must be (n_offsprings, n_matings, n_var).
        n_parents, n_matings, n_var = X.shape
        Y = np.empty((self.n_offsprings, n_matings, n_var), dtype=X.dtype)

        for i in range(n_matings):
            parent_a = self._extract_genome(X[0, i])  # first parent
            parent_b = self._extract_genome(X[1, i]) if n_parents >= 2 else parent_a.copy()

            self._swap_segments(parent_a, parent_b)

            Y[0, i, 0] = parent_a
            Y[1, i, 0] = parent_b

        return Y

    def _swap_segments(self, parent_a: Genome, parent_b: Genome) -> None:
        """Swap segments between parents with configured probability."""
        for seg_a in parent_a.segments:
            if seg_a.key == "role_statement":
                continue
            if random.random() < self.swap_prob:
                seg_b = parent_b.get_segment(seg_a.key)
                if seg_b:
                    seg_a.content, seg_b.content = seg_b.content, seg_a.content


class PromptDuplicateElimination(ElementwiseDuplicateElimination):
    """Detect duplicate genomes using Jaccard similarity on segment content."""

    def __init__(self, threshold: float = 0.9):
        super().__init__()
        self.threshold = threshold

    def is_equal(self, a, b):
        genome_a: Genome = a.X[0]
        genome_b: Genome = b.X[0]

        # Compare all segments via token-level Jaccard
        similarities = []
        for seg_a in genome_a.segments:
            seg_b = genome_b.get_segment(seg_a.key)
            if seg_b:
                tokens_a = set(seg_a.content.lower().split())
                tokens_b = set(seg_b.content.lower().split())
                if tokens_a or tokens_b:
                    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
                    similarities.append(jaccard)

        if not similarities:
            return False
        return (sum(similarities) / len(similarities)) >= self.threshold
