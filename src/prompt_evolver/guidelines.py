"""Guidelines memory: persistent store of discovered prompt improvement patterns."""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Guideline:
    """A single learned guideline."""

    id: str
    text: str
    segment_key: str
    source: str  # "evolution" | "manual" | "feedback"
    keywords: list[str]
    polarity: str  # "positive" | "negative"
    confidence: float
    created: str
    last_seen: str
    times_seen: int


class GuidelinesDB:
    """JSON-file-backed guidelines database."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.guidelines: list[Guideline] = []

    def load(self) -> None:
        """Load guidelines from disk. Creates empty file if missing."""
        if not self.path.exists():
            self.guidelines = []
            return

        with open(self.path) as f:
            data = json.load(f)

        self.guidelines = []
        for g in data:
            self.guidelines.append(
                Guideline(
                    id=g.get("id", str(uuid.uuid4())),
                    text=g["text"],
                    segment_key=g.get("segment_key", ""),
                    source=g.get("source", "manual"),
                    keywords=g.get("keywords", []),
                    polarity=g.get("polarity", "positive"),
                    confidence=g.get("confidence", 0.5),
                    created=g.get("created", ""),
                    last_seen=g.get("last_seen", ""),
                    times_seen=g.get("times_seen", 1),
                )
            )

    def save(self) -> None:
        """Persist guidelines to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump([asdict(g) for g in self.guidelines], f, indent=2)

    def add(
        self,
        text: str,
        segment_key: str,
        source: str = "evolution",
        keywords: list[str] | None = None,
        polarity: str = "positive",
        confidence: float = 0.5,
    ) -> Guideline:
        """Add a guideline, deduplicating against existing entries.

        If a similar guideline exists (>70% word overlap), increments its
        times_seen and updates confidence instead of creating a duplicate.
        """
        now = datetime.now(timezone.utc).isoformat()
        keywords = keywords or []

        # Check for duplicates via normalized word overlap
        new_words = set(text.lower().split())
        for existing in self.guidelines:
            existing_words = set(existing.text.lower().split())
            if not new_words or not existing_words:
                continue
            overlap = len(new_words & existing_words) / len(new_words | existing_words)
            if overlap > 0.7:
                existing.times_seen += 1
                existing.last_seen = now
                existing.confidence = min(1.0, existing.confidence + 0.1)
                return existing

        guideline = Guideline(
            id=str(uuid.uuid4()),
            text=text,
            segment_key=segment_key,
            source=source,
            keywords=keywords,
            polarity=polarity,
            confidence=confidence,
            created=now,
            last_seen=now,
            times_seen=1,
        )
        self.guidelines.append(guideline)
        return guideline

    def get_for_segment(
        self, segment_key: str, min_confidence: float = 0.3
    ) -> list[dict]:
        """Get guidelines for a specific segment, filtered by confidence."""
        return [
            asdict(g)
            for g in self.guidelines
            if g.segment_key == segment_key and g.confidence >= min_confidence
        ]

    def get_random(self, n: int = 1) -> list[dict]:
        """Get n random guidelines."""
        if not self.guidelines:
            return []
        selected = random.sample(self.guidelines, min(n, len(self.guidelines)))
        return [asdict(g) for g in selected]

    def get_all(self) -> list[dict]:
        """Get all guidelines as dicts."""
        return [asdict(g) for g in self.guidelines]

    def export_summary(self) -> str:
        """Human-readable summary of all guidelines, grouped by segment."""
        by_segment: dict[str, list[Guideline]] = {}
        for g in self.guidelines:
            by_segment.setdefault(g.segment_key or "general", []).append(g)

        lines = ["# Guidelines Summary\n"]
        for seg_key in sorted(by_segment.keys()):
            lines.append(f"\n## {seg_key}\n")
            for g in sorted(by_segment[seg_key], key=lambda x: -x.confidence):
                lines.append(
                    f"- [{g.confidence:.1f}] {g.text} "
                    f"(seen {g.times_seen}x, source: {g.source})"
                )

        return "\n".join(lines)
