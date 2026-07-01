from pathlib import Path

from ariostea.adapters.chunk.heading_aware import HeadingAwareChunker
from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser
from ariostea.eval.harness import load_gold

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "eval" / "contextual_corpus"
GOLD = REPO / "eval" / "contextual_gold.json"

# buried target note -> topic words that must NOT appear anywhere in the note
BURIED = {
    "piano.md": ["piano"],
    "chess.md": ["chess"],
    "bicycle.md": ["bicycle", "bike"],
    "guitar.md": ["guitar"],
    "camera.md": ["camera"],
}


def test_every_expected_note_exists_and_is_single():
    for case in load_gold(GOLD):
        assert len(case.expected) == 1, f"{case.query!r} has {len(case.expected)} expected"
        assert (CORPUS / case.expected[0]).exists(), f"missing: {case.expected[0]}"


def test_gold_covers_buried_and_direct_only():
    assert {c.scenario for c in load_gold(GOLD)} == {"buried", "direct"}


def test_buried_notes_never_name_their_topic():
    for name, words in BURIED.items():
        text = (CORPUS / name).read_text(encoding="utf-8").lower()
        for w in words:
            assert w not in text, f"{name} leaks its topic word {w!r}"


def test_buried_targets_chunk_into_at_least_two_chunks():
    parser, chunker = ObsidianMarkdownParser(), HeadingAwareChunker()
    for name in BURIED:
        raw = (CORPUS / name).read_text(encoding="utf-8")
        note, body = parser.parse(name, raw, 0.0)
        assert len(chunker.chunk(note, body)) >= 2, f"{name} did not chunk into >=2"
