"""Tests for fitness evaluation components."""

from unittest.mock import MagicMock

import pytest

from prompt_evolver.config import FitnessConfig, FitnessWeights
from prompt_evolver.fitness import FitnessEvaluator, FitnessResult
from prompt_evolver.genome import Genome, Segment
from prompt_evolver.llm import LLMClient, LLMResponse


def _make_genome(segments: dict[str, str]) -> Genome:
    """Helper to create a Genome from a dict of key -> content."""
    g = Genome()
    for key, content in segments.items():
        g.segments.append(Segment(key=key, header=f"## {key}", content=content, level=2))
    return g


def _make_config(**overrides) -> FitnessConfig:
    return FitnessConfig(
        weights=FitnessWeights(),
        specificity_patterns=overrides.get("specificity_patterns", {
            "guardrails": [r"\b(never|refuse|reject)\b"],
            "expertise_areas": [r"\b[A-Z][A-Za-z]+\b"],
        }),
        calibration_keywords=overrides.get("calibration_keywords", {
            "disposition": ["decision", "judgment"],
            "response_style": ["lead with", "concrete"],
        }),
        guideline_keywords=overrides.get("guideline_keywords", {
            "guardrails": ["never", "push back"],
        }),
        deterministic_floors=overrides.get("deterministic_floors", {
            "structure": 0.6,
            "specificity": 0.3,
        }),
        vague_phrases=["it depends", "best practices suggest"],
        llm_prompts={
            "disposition_rubric": "Rate 1-5: {disposition} vs {scenario}",
            "actionability_rubric": "Count actions: {prompt} for {scenario}",
        },
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock(spec=LLMClient)
    # Default: judge_score returns 4 (good)
    llm.judge_score.return_value = 4
    # Default: complete returns "5" for actionability count
    llm.complete.return_value = LLMResponse(text="5", model="test", usage={})
    return llm


@pytest.fixture
def evaluator(mock_llm):
    config = _make_config()
    return FitnessEvaluator(
        config=config,
        llm=mock_llm,
        segment_keys=["role_statement", "disposition", "expertise_areas",
                       "context_block", "response_style", "strengths",
                       "guardrails", "output_format"],
    )


class TestSpecificity:
    def test_matches_patterns(self, evaluator):
        genome = _make_genome({
            "guardrails": "Never give refunds without approval. Refuse unauthorized requests.",
            "expertise_areas": "Deep experience in SaaS and API troubleshooting.",
        })
        score = evaluator._score_specificity(genome, {"expected_focus": []})
        assert score > 0.0

    def test_penalizes_vagueness(self, evaluator):
        genome = _make_genome({
            "guardrails": "It depends on various factors. Best practices suggest caution.",
            "expertise_areas": "general experience in things.",
        })
        score = evaluator._score_specificity(genome, {"expected_focus": []})
        # Should be lower due to vague phrases
        assert score < 0.5

    def test_expected_focus_matching(self, evaluator):
        genome = _make_genome({
            "role_statement": "Expert in billing disputes and SLA management.",
        })
        scenario = {"expected_focus": ["billing", "SLA"]}
        score = evaluator._score_specificity(genome, scenario)
        assert score > 0.0


class TestStructure:
    def test_all_segments_present(self, evaluator):
        genome = _make_genome({
            "role_statement": "Agent",
            "disposition": "Direct",
            "expertise_areas": "SaaS",
            "context_block": "B2B",
            "response_style": "Lead with action",
            "strengths": "De-escalation",
            "guardrails": "Never over-promise",
            "output_format": "Brief format",
        })
        score = evaluator._score_structure(genome)
        assert score >= 0.8

    def test_missing_segments_penalized(self, evaluator):
        genome = _make_genome({
            "role_statement": "Agent",
            "disposition": "Direct",
        })
        score = evaluator._score_structure(genome)
        assert score < 0.5


class TestCalibration:
    def test_keyword_matching(self, evaluator):
        genome = _make_genome({
            "disposition": "Make clear decisions using professional judgment.",
            "response_style": "Always lead with the concrete recommendation.",
        })
        score = evaluator._score_calibration(genome)
        assert score > 0.0

    def test_no_keywords_neutral(self):
        config = _make_config(calibration_keywords={})
        llm = MagicMock(spec=LLMClient)
        ev = FitnessEvaluator(config=config, llm=llm, segment_keys=["disposition"])
        genome = _make_genome({"disposition": "Some text"})
        score = ev._score_calibration(genome)
        assert score == 0.5  # neutral


class TestDisposition:
    def test_uses_llm_judge(self, evaluator, mock_llm):
        genome = _make_genome({"disposition": "Be direct and decisive."})
        scenario = {"description": "Customer is angry about billing error."}
        score = evaluator._score_disposition(genome, scenario)
        assert mock_llm.judge_score.called
        assert 0.0 <= score <= 1.0

    def test_normalizes_score(self, evaluator, mock_llm):
        mock_llm.judge_score.return_value = 5
        genome = _make_genome({"disposition": "Direct"})
        score = evaluator._score_disposition(genome, {"description": "test"})
        assert score == 1.0

        mock_llm.judge_score.return_value = 1
        score = evaluator._score_disposition(genome, {"description": "test"})
        assert score == 0.0


class TestFitnessResult:
    def test_as_tuple(self):
        r = FitnessResult(0.8, 0.9, 0.7, 0.6, 0.5, 0.4)
        assert r.as_tuple() == (0.8, 0.9, 0.7, 0.6, 0.5, 0.4)

    def test_passes_floors(self):
        r = FitnessResult(specificity=0.5, structure=0.8)
        assert r.passes_floors({"specificity": 0.3, "structure": 0.6})

    def test_fails_floors(self):
        r = FitnessResult(specificity=0.2, structure=0.8)
        assert not r.passes_floors({"specificity": 0.3})
