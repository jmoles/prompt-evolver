"""Genome representation: parse and render structured markdown agent prompts."""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field

from prompt_evolver.config import SegmentDef


@dataclass
class Segment:
    """A single section of an agent prompt."""

    key: str
    header: str  # the actual header text found (e.g., "## Your disposition")
    content: str  # everything between this header and the next
    level: int  # header level (1 for #, 2 for ##, 3 for ###)


@dataclass
class Genome:
    """A structured agent prompt decomposed into evolvable segments."""

    segments: list[Segment] = field(default_factory=list)
    raw_header: str = ""  # the # Title line + opening paragraph

    def to_markdown(self) -> str:
        """Render genome back to markdown string."""
        return render_markdown(self)

    def get_segment(self, key: str) -> Segment | None:
        """Get a segment by key."""
        for s in self.segments:
            if s.key == key:
                return s
        return None

    def set_segment(self, key: str, content: str) -> None:
        """Update a segment's content by key."""
        for s in self.segments:
            if s.key == key:
                s.content = content
                return
        raise KeyError(f"Segment not found: {key}")

    def segment_keys(self) -> list[str]:
        """Return list of segment keys in order."""
        return [s.key for s in self.segments]

    def copy(self) -> Genome:
        """Deep copy this genome."""
        return copy.deepcopy(self)


_HEADER_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)


def _match_header(header_text: str, segment_defs: list[SegmentDef]) -> str | None:
    """Match a header string to a segment key using configured variants.

    Uses case-insensitive substring matching against header_variants.
    Returns the segment key or None if no match.
    """
    header_lower = header_text.lower().strip()
    for sdef in segment_defs:
        for variant in sdef.header_variants:
            if variant.lower() in header_lower:
                return sdef.key
    return None


def parse_markdown(text: str, segment_defs: list[SegmentDef]) -> Genome:
    """Parse a structured markdown prompt into a Genome.

    The role_statement segment is everything before the first ## header.
    Subsequent sections are matched to segment definitions by header text.
    Unrecognized headers are folded into the preceding segment.

    Args:
        text: The full markdown text of an agent prompt.
        segment_defs: Segment definitions from config.

    Returns:
        A Genome with matched segments.
    """
    lines = text.split("\n")
    genome = Genome()

    # Find ## and ### headers (level 2+) as section boundaries.
    # Level 1 (#) headers are treated as part of the role_statement.
    headers: list[tuple[int, int, str]] = []  # (line_idx, level, header_text)
    for i, line in enumerate(lines):
        m = _HEADER_RE.match(line)
        if m:
            level = len(m.group(1))
            if level >= 2:  # only ## and ### are section boundaries
                header_text = m.group(2).strip()
                headers.append((i, level, header_text))

    if not headers:
        # No headers found — entire text is the role_statement
        genome.raw_header = text
        role_def = next((s for s in segment_defs if s.key == "role_statement"), None)
        if role_def:
            genome.segments.append(
                Segment(key="role_statement", header="", content=text.strip(), level=0)
            )
        return genome

    # Everything before the first ## header is the role_statement
    first_header_line = headers[0][0]
    raw_header = "\n".join(lines[:first_header_line]).strip()
    genome.raw_header = raw_header

    # Add role_statement segment if configured
    role_def = next((s for s in segment_defs if s.key == "role_statement"), None)
    if role_def and raw_header:
        genome.segments.append(
            Segment(key="role_statement", header="", content=raw_header, level=0)
        )

    # Process each header section
    matched_keys: set[str] = set()
    current_segment: Segment | None = None

    for idx, (line_idx, level, header_text) in enumerate(headers):
        # Determine content end (next header or end of file)
        if idx + 1 < len(headers):
            end_line = headers[idx + 1][0]
        else:
            end_line = len(lines)

        content = "\n".join(lines[line_idx + 1 : end_line]).strip()
        full_header = lines[line_idx].strip()

        # Try to match this header to a segment definition
        matched_key = _match_header(header_text, segment_defs)

        if matched_key and matched_key not in matched_keys:
            # New matched segment
            current_segment = Segment(
                key=matched_key,
                header=full_header,
                content=content,
                level=level,
            )
            genome.segments.append(current_segment)
            matched_keys.add(matched_key)
        elif current_segment is not None:
            # Unrecognized header — fold into preceding segment
            current_segment.content += "\n\n" + full_header + "\n" + content
        else:
            # Unrecognized header before any matched segment — fold into raw_header
            genome.raw_header += "\n\n" + full_header + "\n" + content

    return genome


def render_markdown(genome: Genome) -> str:
    """Render a Genome back to a markdown string.

    Reconstructs the original markdown structure from segments.

    Args:
        genome: The genome to render.

    Returns:
        Complete markdown string.
    """
    parts: list[str] = []

    # Find the role_statement segment
    role_seg = genome.get_segment("role_statement")

    if role_seg:
        parts.append(role_seg.content)
    elif genome.raw_header:
        parts.append(genome.raw_header)

    for seg in genome.segments:
        if seg.key == "role_statement":
            continue
        if seg.header:
            parts.append(seg.header)
        if seg.content:
            parts.append(seg.content)

    return "\n\n".join(parts) + "\n"
