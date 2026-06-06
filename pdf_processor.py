"""
pdf_processor.py
----------------
Handles PDF ingestion, text extraction, and section segmentation.
Uses PyMuPDF (fitz) for fast, Apple Silicon-compatible extraction.
"""

import re
import fitz  # PyMuPDF
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class Section:
    """Represents a logical section from the PDF."""
    title: str
    content: str
    page_start: int
    page_end: int
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.content.split())


@dataclass
class ProcessedPDF:
    """Complete output from the PDF processor."""
    file_path: str
    title: str
    full_text: str
    sections: List[Section]
    total_pages: int
    total_words: int = 0

    def __post_init__(self):
        self.total_words = len(self.full_text.split())


def extract_text_from_pdf(pdf_path: str) -> ProcessedPDF:
    """
    Extract and clean text from a PDF file.
    Segments into logical sections based on heading patterns.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"File must be a PDF: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # --- Extract raw text per page ---
    raw_pages: List[str] = []
    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text")
        raw_pages.append(text)

    doc.close()

    # --- Clean and normalize text ---
    full_text = _clean_text("\n".join(raw_pages))

    # --- Segment into logical sections ---
    sections = _segment_into_sections(raw_pages)

    # Detect document title (first non-empty short line)
    title = _detect_title(raw_pages, path.stem)

    return ProcessedPDF(
        file_path=str(path.resolve()),
        title=title,
        full_text=full_text,
        sections=sections,
        total_pages=total_pages,
    )


def _clean_text(raw: str) -> str:
    """Remove noise: headers/footers patterns, hyphenation, extra whitespace."""
    # Rejoin hyphenated line breaks
    text = re.sub(r"-\n(\w)", r"\1", raw)
    # Replace form feeds and multiple newlines
    text = re.sub(r"\f", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Remove lines that look like page numbers (just digits)
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def _is_heading(line: str) -> bool:
    """
    Heuristic to detect section headings.
    Matches: numbered headings, ALL CAPS short lines, Title Case lines.
    """
    line = line.strip()
    if not line or len(line) > 120:
        return False

    # Numbered heading: "1. Introduction", "2.3 Key Concepts"
    if re.match(r"^\d+[\.\d]*\s+[A-Z]", line):
        return True
    # ALL CAPS heading (short)
    if line.isupper() and 3 <= len(line.split()) <= 10:
        return True
    # Title Case heading (most words capitalized, short line)
    words = line.split()
    if (
        2 <= len(words) <= 8
        and sum(1 for w in words if w and w[0].isupper()) / len(words) > 0.7
        and not line.endswith(".")
    ):
        return True
    return False


def _segment_into_sections(raw_pages: List[str]) -> List[Section]:
    """
    Split the document into logical sections by detecting headings.
    Falls back to fixed-size chunks if no headings found.
    """
    sections: List[Section] = []
    current_title = "Introduction"
    current_lines: List[str] = []
    current_page_start = 0

    for page_num, page_text in enumerate(raw_pages):
        for line in page_text.splitlines():
            clean_line = line.strip()
            if _is_heading(clean_line) and len(current_lines) > 10:
                # Save previous section
                content = _clean_text("\n".join(current_lines))
                if content.strip():
                    sections.append(Section(
                        title=current_title,
                        content=content,
                        page_start=current_page_start,
                        page_end=page_num,
                    ))
                current_title = clean_line
                current_lines = []
                current_page_start = page_num
            else:
                current_lines.append(clean_line)

    # Add last section
    if current_lines:
        content = _clean_text("\n".join(current_lines))
        if content.strip():
            sections.append(Section(
                title=current_title,
                content=content,
                page_start=current_page_start,
                page_end=len(raw_pages) - 1,
            ))

    # Fallback: if fewer than 2 sections found, chunk by word count
    if len(sections) < 2:
        sections = _chunk_by_words(
            _clean_text("\n".join(raw_pages)),
            chunk_words=400,
        )

    return sections


def _chunk_by_words(text: str, chunk_words: int = 400) -> List[Section]:
    """Fallback: split text into roughly equal word-count chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_words):
        chunk_text = " ".join(words[i: i + chunk_words])
        chunk_num = i // chunk_words + 1
        chunks.append(Section(
            title=f"Section {chunk_num}",
            content=chunk_text,
            page_start=0,
            page_end=0,
        ))
    return chunks


def _detect_title(raw_pages: List[str], fallback: str) -> str:
    """Try to detect the document title from the first page."""
    if not raw_pages:
        return fallback
    for line in raw_pages[0].splitlines():
        line = line.strip()
        if 3 <= len(line.split()) <= 12 and line[0].isupper() if line else False:
            return line
    return fallback.replace("_", " ").title()
