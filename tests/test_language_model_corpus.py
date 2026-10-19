import hashlib
import json
from pathlib import Path

import pytest

from ai_platform.machine_learning.language_model.corpus import (
    build_corpus,
    clean_project_gutenberg_text,
    sha256_text,
)


def _make_source(
    tmp_path,
    source_id,
    text,
    *,
    split="train",
    rights_status="research_only",
):
    source_path = tmp_path / f"{source_id}.txt"
    source_path.write_text(text, encoding="utf-8")
    return {
        "source_id": source_id,
        "path": str(source_path),
        "title": f"Title {source_id}",
        "author": "Test Author",
        "source_url": f"https://example.org/{source_id}",
        "license_name": "Test license; rights status is illustrative",
        "license_url": "https://example.org/license",
        "rights_status": rights_status,
        "split": split,
    }


def test_gutenberg_cleaner_removes_header_and_footer():
    raw = (
        "Header material\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "\nMain body.\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\n"
        "Footer material\n"
    )

    cleaned = clean_project_gutenberg_text(raw)

    assert cleaned == "Main body.\n"
    assert "Header material" not in cleaned
    assert "Footer material" not in cleaned


def test_gutenberg_cleaner_rejects_one_missing_boundary_marker():
    with pytest.raises(ValueError, match="Only one"):
        clean_project_gutenberg_text(
            "*** START OF THE PROJECT GUTENBERG EBOOK SAMPLE ***\nText"
        )


def test_build_corpus_includes_train_and_excludes_test(tmp_path):
    train = _make_source(
        tmp_path, "alice", "Alice text.\n", split="train"
    )
    test = _make_source(
        tmp_path, "sherlock", "Sherlock text.\n", split="test"
    )

    output = tmp_path / "prepared" / "train.txt"
    manifest_path = tmp_path / "prepared" / "manifest.json"

    manifest = build_corpus(
        [train, test],
        output,
        manifest_path,
        target_split="train",
        purpose="research",
    )

    text = output.read_text(encoding="utf-8")
    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "Alice text." in text
    assert "Sherlock text." not in text
    assert manifest["source_count"] == 1
    assert saved_manifest["sources"][0]["source_id"] == "alice"
    assert saved_manifest["excluded_sources"][0]["source_id"] == "sherlock"


def test_manifest_hash_matches_written_corpus(tmp_path):
    source = _make_source(tmp_path, "sample", "Example text.\n")
    output = tmp_path / "corpus.txt"
    manifest_path = tmp_path / "manifest.json"

    manifest = build_corpus([source], output, manifest_path)
    written = output.read_text(encoding="utf-8")

    assert manifest["corpus_sha256"] == sha256_text(written)
    assert manifest["sources"][0]["_raw_sha256"] == hashlib.sha256(
        Path(source["path"]).read_bytes()
    ).hexdigest()


def test_pending_review_source_cannot_be_included(tmp_path):
    source = _make_source(
        tmp_path, "pending", "Some text.", rights_status="pending_review"
    )

    with pytest.raises(ValueError, match="pending rights review"):
        build_corpus(
            [source],
            tmp_path / "corpus.txt",
            tmp_path / "manifest.json",
        )


def test_research_only_source_is_rejected_for_commercial_purpose(tmp_path):
    source = _make_source(
        tmp_path, "research", "Some text.", rights_status="research_only"
    )

    with pytest.raises(ValueError, match="not marked approved"):
        build_corpus(
            [source],
            tmp_path / "corpus.txt",
            tmp_path / "manifest.json",
            purpose="commercial",
        )


def test_duplicate_cleaned_text_across_splits_is_rejected(tmp_path):
    train = _make_source(
        tmp_path, "train-copy", "Same text.\n", split="train"
    )
    test = _make_source(
        tmp_path, "test-copy", "Same text.\n", split="test"
    )

    with pytest.raises(ValueError, match="Duplicate cleaned content"):
        build_corpus(
            [train, test],
            tmp_path / "corpus.txt",
            tmp_path / "manifest.json",
        )


def test_build_corpus_rejects_missing_target_split(tmp_path):
    source = _make_source(
        tmp_path, "heldout", "Held-out text.", split="test"
    )

    with pytest.raises(ValueError, match="No eligible sources"):
        build_corpus(
            [source],
            tmp_path / "train.txt",
            tmp_path / "manifest.json",
            target_split="train",
        )
