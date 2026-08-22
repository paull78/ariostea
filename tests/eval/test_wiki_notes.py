from pathlib import Path

import pytest

from ariostea.eval.wiki_notes import cluster_of, load_corpus_notes, note_titles, strip_frontmatter

NOTE = """---
title: Violin
lang: en
cluster: string-instruments
revid: 12345
source: https://en.wikipedia.org/w/index.php?title=Violin&oldid=12345
license: CC BY-SA 4.0
---

# Violin

The violin is a wooden chordophone.
"""


def test_strip_frontmatter_removes_the_block_and_leading_blank_lines():
    assert strip_frontmatter(NOTE).startswith("# Violin")


def test_strip_frontmatter_leaves_a_note_without_frontmatter_alone():
    assert strip_frontmatter("# Violin\n\nBody.\n") == "# Violin\n\nBody.\n"


def test_strip_frontmatter_leaves_an_unterminated_block_alone():
    # A truncated file must not have its whole body silently eaten.
    raw = "---\ntitle: Violin\n\n# Violin\n"
    assert strip_frontmatter(raw) == raw


def test_strip_frontmatter_handles_an_empty_block():
    assert strip_frontmatter("---\n---\n\n# Violin\n") == "# Violin\n"


def test_load_corpus_notes_keys_by_cluster_relative_path(tmp_path: Path):
    (tmp_path / "string-instruments").mkdir()
    (tmp_path / "string-instruments" / "violin.md").write_text(NOTE, encoding="utf-8")
    notes = load_corpus_notes(tmp_path)
    assert list(notes) == ["string-instruments/violin.md"]
    assert notes["string-instruments/violin.md"].startswith("# Violin")


def test_load_corpus_notes_returns_paths_in_sorted_order(tmp_path: Path):
    for cluster, slug in (("strings", "violin"), ("cheese", "brie"), ("strings", "cello")):
        (tmp_path / cluster).mkdir(exist_ok=True)
        (tmp_path / cluster / f"{slug}.md").write_text(NOTE, encoding="utf-8")
    assert list(load_corpus_notes(tmp_path)) == [
        "cheese/brie.md",
        "strings/cello.md",
        "strings/violin.md",
    ]


def test_note_titles_reads_the_h1():
    assert note_titles({"a/violin.md": "# Violin\n\nBody."}) == {"a/violin.md": "Violin"}


def test_note_titles_rejects_a_note_with_no_h1():
    # Every corpus note is rendered with an H1; a missing one means the file is
    # damaged, and a silent path-stem fallback would hide that from the gate
    # that uses titles to reject title-answerable queries.
    with pytest.raises(ValueError, match="no H1"):
        note_titles({"a/violin.md": "Body with no heading."})


def test_note_titles_ignores_a_deeper_heading():
    assert note_titles({"a/x.md": "## Section\n\n# Violin\n"}) == {"a/x.md": "Violin"}


def test_cluster_of_takes_the_first_path_segment():
    assert cluster_of("string-instruments/violin.md") == "string-instruments"
