"""Scenario extraction from PDFs and curation for fitness evaluation."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path

from prompt_evolver.llm import LLMClient


@dataclass
class Scenario:
    """A test scenario for evaluating agent prompts."""

    id: str
    title: str
    description: str
    expected_focus: list[str]
    relevant_segments: list[str]
    difficulty: str  # "basic" | "intermediate" | "adversarial"
    source: str


_CURATION_PROMPT = """Given these document excerpts, extract realistic professional scenarios
that would test an AI agent's expertise and judgment. Each scenario should present a
situation where the agent needs to apply its specialized knowledge.

For each scenario, provide:
- title: brief descriptive title
- description: 2-3 sentence scenario with enough context for evaluation
- expected_focus: list of 3-5 topics a good response should address
- relevant_segments: which of these prompt sections are most tested: {segment_keys}
- difficulty: "basic", "intermediate", or "adversarial"

DOCUMENT EXCERPTS:
{text}

Extract {n_scenarios} diverse scenarios covering different difficulty levels.
Output as a JSON array of objects with the fields above."""


class ScenarioExtractor:
    """Extract and curate scenarios from source documents."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract_from_pdf(self, pdf_path: str) -> list[str]:
        """Extract text chunks from a PDF file.

        Uses pypdf for text extraction. Returns a list of text passages,
        one per page.
        """
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        chunks = []
        for page in reader.pages:
            text = page.extract_text()
            if text and len(text.strip()) > 100:
                chunks.append(text.strip())
        return chunks

    def curate_scenarios(
        self,
        raw_texts: list[str],
        segment_keys: list[str],
        n_scenarios: int = 20,
    ) -> list[Scenario]:
        """Use LLM to curate test scenarios from raw text.

        Args:
            raw_texts: Extracted text passages from source documents.
            segment_keys: Names of prompt segments (for relevance tagging).
            n_scenarios: Number of scenarios to generate.

        Returns:
            List of curated Scenario objects.
        """
        # Combine text chunks, truncating to fit context
        combined = "\n\n---\n\n".join(raw_texts)
        if len(combined) > 8000:
            combined = combined[:8000] + "\n\n[truncated]"

        prompt = _CURATION_PROMPT.format(
            segment_keys=", ".join(segment_keys),
            text=combined,
            n_scenarios=n_scenarios,
        )

        data = self.llm.complete_json(user=prompt)

        scenarios = []
        items = data if isinstance(data, list) else data.get("scenarios", [])
        for item in items:
            scenarios.append(
                Scenario(
                    id=str(uuid.uuid4()),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    expected_focus=item.get("expected_focus", []),
                    relevant_segments=item.get("relevant_segments", []),
                    difficulty=item.get("difficulty", "intermediate"),
                    source="curated",
                )
            )

        return scenarios

    def save_scenarios(self, scenarios: list[Scenario], output_path: str) -> None:
        """Save scenarios to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(s) for s in scenarios], f, indent=2)

    def load_scenarios(self, path: str) -> list[dict]:
        """Load scenarios from a JSON file."""
        with open(path) as f:
            return json.load(f)
