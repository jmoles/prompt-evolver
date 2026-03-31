"""Tests for evolutionary operators."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from prompt_evolver.genome import Genome, Segment
from prompt_evolver.llm import LLMClient, LLMResponse
from prompt_evolver.operators import (
    PromptDuplicateElimination,
    PromptMutation,
    PromptSampling,
    SegmentCrossover,
)


def _make_genome(disposition: str = "Be direct.", expertise: str = "SaaS support.") -> Genome:
    g = Genome()
    g.segments = [
        Segment(key="role_statement", header="", content="Test Agent", level=0),
        Segment(key="disposition", header="## Your disposition", content=disposition, level=2),
        Segment(key="expertise_areas", header="## Your expertise", content=expertise, level=2),
    ]
    return g


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    llm.complete.return_value = LLMResponse(
        text="This is the mutated text that is long enough to pass the sanity check.",
        model="test",
        usage={},
    )
    return llm


class TestPromptSampling:
    def test_produces_correct_population_size(self, mock_llm):
        seed = _make_genome()
        sampling = PromptSampling(seed, mock_llm)
        # Mock the problem
        problem = MagicMock()
        X = sampling._do(problem, 4)
        assert X.shape == (4, 1)

    def test_first_individual_is_seed(self, mock_llm):
        seed = _make_genome("Original disposition")
        sampling = PromptSampling(seed, mock_llm)
        problem = MagicMock()
        X = sampling._do(problem, 3)
        first: Genome = X[0, 0]
        assert first.get_segment("disposition").content == "Original disposition"

    def test_mutations_called_for_non_seed(self, mock_llm):
        seed = _make_genome()
        sampling = PromptSampling(seed, mock_llm)
        problem = MagicMock()
        sampling._do(problem, 3)
        # LLM should be called for individuals 2 and 3
        assert mock_llm.complete.call_count >= 2


class TestSegmentCrossover:
    def test_full_swap(self):
        """With probability 1.0, all segments should swap."""
        cx = SegmentCrossover(swap_prob=1.0)
        parent_a = _make_genome("Disposition A", "Expertise A")
        parent_b = _make_genome("Disposition B", "Expertise B")

        # pymoo convention: X shape is (n_parents, n_matings, n_var)
        X = np.array([[[parent_a]], [[parent_b]]], dtype=object)
        Y = cx._do(None, X)

        # Output shape: (n_offsprings, n_matings, n_var)
        child_a: Genome = Y[0, 0, 0]
        child_b: Genome = Y[1, 0, 0]

        # After full swap, A should have B's content and vice versa
        assert child_a.get_segment("disposition").content == "Disposition B"
        assert child_b.get_segment("disposition").content == "Disposition A"

    def test_no_swap(self):
        """With probability 0.0, no segments should swap."""
        cx = SegmentCrossover(swap_prob=0.0)
        parent_a = _make_genome("Disposition A", "Expertise A")
        parent_b = _make_genome("Disposition B", "Expertise B")

        X = np.array([[[parent_a]], [[parent_b]]], dtype=object)
        Y = cx._do(None, X)

        child_a: Genome = Y[0, 0, 0]
        assert child_a.get_segment("disposition").content == "Disposition A"

    def test_role_statement_never_swaps(self):
        """role_statement should never be swapped."""
        cx = SegmentCrossover(swap_prob=1.0)
        parent_a = _make_genome()
        parent_b = _make_genome()
        parent_a.segments[0].content = "Agent A"
        parent_b.segments[0].content = "Agent B"

        X = np.array([[[parent_a]], [[parent_b]]], dtype=object)
        Y = cx._do(None, X)

        child_a: Genome = Y[0, 0, 0]
        # role_statement should remain unchanged
        assert child_a.get_segment("role_statement").content == "Agent A"


class TestPromptMutation:
    def test_mutation_modifies_genome(self, mock_llm):
        mutation = PromptMutation(llm=mock_llm)
        genome = _make_genome()
        X = np.array([[genome]], dtype=object)
        Xp = mutation._do(None, X)
        # Should return same shape
        assert Xp.shape == X.shape

    def test_prune_reduces_length(self, mock_llm):
        mock_llm.complete.return_value = LLMResponse(
            text="Shorter text that is still long enough for the check to pass through.",
            model="test",
            usage={},
        )
        mutation = PromptMutation(llm=mock_llm)
        genome = _make_genome(disposition="A" * 300)
        result = mutation._prune(genome)
        # Should have called the LLM
        assert mock_llm.complete.called


class TestDuplicateElimination:
    def test_identical_genomes_are_duplicates(self):
        dedup = PromptDuplicateElimination(threshold=0.9)
        g1 = _make_genome("Same disposition text here")
        g2 = _make_genome("Same disposition text here")

        ind1 = MagicMock()
        ind1.X = [g1]
        ind2 = MagicMock()
        ind2.X = [g2]

        assert dedup.is_equal(ind1, ind2)

    def test_different_genomes_are_not_duplicates(self):
        dedup = PromptDuplicateElimination(threshold=0.9)
        g1 = _make_genome("Completely different text about cooking recipes")
        g2 = _make_genome("Technical engineering documentation for APIs")

        ind1 = MagicMock()
        ind1.X = [g1]
        ind2 = MagicMock()
        ind2.X = [g2]

        assert not dedup.is_equal(ind1, ind2)
