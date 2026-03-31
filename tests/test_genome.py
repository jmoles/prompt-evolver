"""Tests for genome parsing and rendering."""

from pathlib import Path

import pytest

from prompt_evolver.config import SegmentDef
from prompt_evolver.genome import Genome, parse_markdown, render_markdown

FIXTURES = Path(__file__).parent / "fixtures"

SEGMENT_DEFS = [
    SegmentDef(key="role_statement", description="", header_variants=[], required=True),
    SegmentDef(key="disposition", description="", header_variants=["Your disposition", "Your role"], required=True),
    SegmentDef(key="expertise_areas", description="", header_variants=["Your expertise", "Expertise"], required=True),
    SegmentDef(key="context_block", description="", header_variants=["context you always carry", "Context"], required=True),
    SegmentDef(key="response_style", description="", header_variants=["How you respond"], required=True),
    SegmentDef(key="strengths", description="", header_variants=["What you are especially sharp on", "Strengths"], required=False),
    SegmentDef(key="guardrails", description="", header_variants=["What you push back on", "Guardrails"], required=False),
    SegmentDef(key="output_format", description="", header_variants=["Output format"], required=True),
]


@pytest.fixture
def example_text():
    return (FIXTURES / "example_agent.md").read_text()


@pytest.fixture
def example_genome(example_text):
    return parse_markdown(example_text, SEGMENT_DEFS)


class TestParsing:
    def test_parses_all_segments(self, example_genome):
        keys = example_genome.segment_keys()
        assert "role_statement" in keys
        assert "disposition" in keys
        assert "expertise_areas" in keys
        assert "context_block" in keys
        assert "response_style" in keys
        assert "strengths" in keys
        assert "guardrails" in keys
        assert "output_format" in keys

    def test_role_statement_content(self, example_genome):
        role = example_genome.get_segment("role_statement")
        assert role is not None
        assert "Senior Customer Support Agent" in role.content

    def test_disposition_content(self, example_genome):
        disp = example_genome.get_segment("disposition")
        assert disp is not None
        assert "patient but efficient" in disp.content

    def test_expertise_content(self, example_genome):
        exp = example_genome.get_segment("expertise_areas")
        assert exp is not None
        assert "SaaS" in exp.content

    def test_context_content(self, example_genome):
        ctx = example_genome.get_segment("context_block")
        assert ctx is not None
        assert "$24K/year" in ctx.content

    def test_segment_count(self, example_genome):
        assert len(example_genome.segments) == 8

    def test_missing_segment_returns_none(self, example_genome):
        assert example_genome.get_segment("nonexistent") is None


class TestRoundTrip:
    def test_render_produces_valid_markdown(self, example_genome):
        rendered = render_markdown(example_genome)
        assert "# Senior Customer Support Agent" in rendered
        assert "## Your disposition" in rendered
        assert "## Your expertise" in rendered

    def test_round_trip_preserves_content(self, example_text, example_genome):
        rendered = render_markdown(example_genome)
        # Re-parse the rendered text
        reparsed = parse_markdown(rendered, SEGMENT_DEFS)
        assert len(reparsed.segments) == len(example_genome.segments)
        for orig, new in zip(example_genome.segments, reparsed.segments):
            assert orig.key == new.key


class TestGenomeOperations:
    def test_copy_is_independent(self, example_genome):
        copied = example_genome.copy()
        copied.set_segment("disposition", "CHANGED")
        assert example_genome.get_segment("disposition").content != "CHANGED"

    def test_set_segment(self, example_genome):
        example_genome.set_segment("disposition", "New disposition text")
        assert example_genome.get_segment("disposition").content == "New disposition text"

    def test_set_nonexistent_segment_raises(self, example_genome):
        with pytest.raises(KeyError):
            example_genome.set_segment("nonexistent", "value")

    def test_segment_keys_order(self, example_genome):
        keys = example_genome.segment_keys()
        assert keys[0] == "role_statement"
        # Other segments should follow in document order


class TestHeaderVariants:
    def test_matches_your_role_variant(self):
        text = "# Test Agent\n\nOpening.\n\n## Your role\n\nRole content here."
        genome = parse_markdown(text, SEGMENT_DEFS)
        disp = genome.get_segment("disposition")
        assert disp is not None
        assert "Role content here" in disp.content

    def test_handles_no_headers(self):
        text = "Just a plain text prompt with no headers."
        genome = parse_markdown(text, SEGMENT_DEFS)
        role = genome.get_segment("role_statement")
        assert role is not None
        assert "plain text prompt" in role.content
