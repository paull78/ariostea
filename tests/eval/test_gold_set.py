from pathlib import Path

from ariostea.eval.harness import load_gold

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "eval" / "corpus"
GOLD = REPO / "eval" / "gold.json"


def test_every_expected_note_exists_and_is_single():
    for case in load_gold(GOLD):
        # single-correct-note assumption holds
        assert len(case.expected) == 1, (
            f"case {case.query!r} has {len(case.expected)} expected notes"
        )
        assert (CORPUS / case.expected[0]).exists(), f"missing corpus note: {case.expected[0]}"


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
