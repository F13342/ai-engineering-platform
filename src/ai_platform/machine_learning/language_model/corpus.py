"""Source-aware corpus preparation with split and rights-status safeguards."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence


_ALLOWED_SPLITS = {"train", "validation", "test"}
_ALLOWED_RIGHTS_STATUSES = {"approved", "research_only", "pending_review"}


def sha256_text(text: str) -> str:
    """Return the SHA-256 digest of UTF-8 encoded text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_project_gutenberg_text(raw_text: str) -> str:
    """Extract a Gutenberg book body when its standard boundary markers exist.

    If neither marker is present, treat the input as an already-extracted text.
    If only one marker is present, reject the input instead of silently keeping
    a potentially contaminated header or footer.
    """
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a string.")

    text = unicodedata.normalize(
        "NFC",
        raw_text.replace("\r\n", "\n").replace("\r", "\n"),
    )

    start_match = re.search(
        r"\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*?\*{3}",
        text,
        flags=re.IGNORECASE,
    )
    end_match = re.search(
        r"\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK[^\n]*?\*{3}",
        text,
        flags=re.IGNORECASE,
    )

    if bool(start_match) != bool(end_match):
        raise ValueError(
            "Only one Project Gutenberg boundary marker was found; "
            "check the source file before cleaning it."
        )

    if start_match and end_match:
        if start_match.end() >= end_match.start():
            raise ValueError("Project Gutenberg boundary markers are out of order.")
        text = text[start_match.end():end_match.start()]

    text = "\n".join(line.rstrip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if not text:
        raise ValueError("The cleaned text is empty.")

    return text + "\n"


def build_corpus(
    sources: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    target_split: str = "train",
    purpose: str = "research",
) -> dict[str, Any]:
    """Build one split-specific text corpus from source records.

    Required source fields:
      source_id, path, title, author, source_url, license_name, license_url,
      rights_status, split.

    rights_status is a project workflow decision, not a legal determination.
    Use "approved" only after the responsible reviewer has approved that use.
    """
    if target_split not in _ALLOWED_SPLITS:
        raise ValueError(f"target_split must be one of {sorted(_ALLOWED_SPLITS)}.")
    if purpose not in {"research", "commercial"}:
        raise ValueError("purpose must be 'research' or 'commercial'.")
    if not sources:
        raise ValueError("At least one source record is required.")

    required_fields = (
        "source_id", "path", "title", "author", "source_url",
        "license_name", "license_url", "rights_status", "split",
    )

    processed: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}

    # Inspect every source so that content duplicated across splits is rejected.
    for original in sources:
        missing = [key for key in required_fields if not original.get(key)]
        if missing:
            raise ValueError(
                f"Source record is missing required fields: {', '.join(missing)}."
            )

        source = dict(original)
        source_id = str(source["source_id"])
        if source_id in seen_ids:
            raise ValueError(f"Duplicate source_id: {source_id}")
        seen_ids.add(source_id)

        split = str(source["split"])
        if split not in _ALLOWED_SPLITS:
            raise ValueError(f"Invalid split for {source_id}: {split}")

        rights_status = str(source["rights_status"])
        if rights_status not in _ALLOWED_RIGHTS_STATUSES:
            raise ValueError(
                f"Invalid rights_status for {source_id}: {rights_status}"
            )

        source_path = Path(str(source["path"])).expanduser()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Source file for {source_id} was not found: {source_path}"
            )

        source_bytes = source_path.read_bytes()
        raw_text = source_bytes.decode("utf-8")
        cleaned_text = clean_project_gutenberg_text(raw_text)
        content_hash = sha256_text(cleaned_text)
        raw_hash = hashlib.sha256(source_bytes).hexdigest()

        previous_source_id = seen_hashes.get(content_hash)
        if previous_source_id is not None:
            raise ValueError(
                "Duplicate cleaned content detected across source records: "
                f"{previous_source_id} and {source_id}."
            )
        seen_hashes[content_hash] = source_id

        source["path"] = str(source_path.resolve())
        source["_cleaned_text"] = cleaned_text
        source["_raw_sha256"] = raw_hash
        source["_cleaned_sha256"] = content_hash
        source["_cleaned_characters"] = len(cleaned_text)
        processed.append(source)

    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []

    for source in processed:
        if source["split"] != target_split:
            excluded.append({
                "source_id": str(source["source_id"]),
                "split": str(source["split"]),
                "reason": "Different split from requested target_split.",
            })
            continue

        rights_status = source["rights_status"]
        if rights_status == "pending_review":
            raise ValueError(
                f"Source {source['source_id']} is pending rights review "
                "and cannot be included."
            )
        if purpose == "commercial" and rights_status != "approved":
            raise ValueError(
                f"Source {source['source_id']} is not marked approved "
                "for this project's commercial-use workflow."
            )

        selected.append(source)

    if not selected:
        raise ValueError(f"No eligible sources found for split '{target_split}'.")

    corpus_text = "\n\n".join(
        source["_cleaned_text"].strip() for source in selected
    ) + "\n"

    destination = Path(output_path)
    manifest_destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(corpus_text, encoding="utf-8")

    def public_record(source: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: source[key]
            for key in (
                "source_id", "title", "author", "source_url",
                "license_name", "license_url", "rights_status", "split",
                "path", "_raw_sha256", "_cleaned_sha256",
                "_cleaned_characters",
            )
        }

    manifest: dict[str, Any] = {
        "format_version": 1,
        "purpose": purpose,
        "target_split": target_split,
        "output_path": str(destination.resolve()),
        "source_count": len(selected),
        "character_count": len(corpus_text),
        "corpus_sha256": sha256_text(corpus_text),
        "sources": [public_record(source) for source in selected],
        "excluded_sources": excluded,
        "notes": [
            "Rights status is a project review label, not legal advice.",
            "Keep source licenses, terms, and attribution information with artifacts.",
            "Do not assume that training data licensing alone settles model-output rights.",
        ],
    }

    manifest_destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    return manifest
