from pathlib import Path

from ariostea.eval.harness import load_gold

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "eval" / "corpus"
GOLD = REPO / "eval" / "gold.json"


def test_every_expected_note_exists_and_is_single():
    for case in load_gold(GOLD):
        assert len(case.expected) == 1  # single-correct-note assumption holds
        assert (CORPUS / case.expected[0]).exists()


def test_gold_covers_all_scenarios():
    scenarios = {c.scenario for c in load_gold(GOLD)}
    assert scenarios == {
        "same",
        "en→it",
        "es→it",
        "it→en",
        "es→en",
        "en→es",
        "it→es",
        "accent",
        "inflection",
    }
