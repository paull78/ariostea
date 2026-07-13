import json
from pathlib import Path

from ariostea.eval.wiki_gold import AnswerSpan, WikiGoldCase, load_wiki_gold, validate_wiki_gold


def test_load_wiki_gold_parses_spans(tmp_path):
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            [
                {
                    "query": "how is a violin tuned",
                    "query_lang": "en",
                    "type": "buried",
                    "scenario": "buried",
                    "expected_notes": ["string-instruments/violin.md"],
                    "answer_spans": [
                        {"note": "string-instruments/violin.md", "text": "tuned in perfect fifths"}
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    cases = load_wiki_gold(path)

    assert cases == [
        WikiGoldCase(
            query="how is a violin tuned",
            query_lang="en",
            type="buried",
            scenario="buried",
            expected_notes=("string-instruments/violin.md",),
            answer_spans=(
                AnswerSpan(note="string-instruments/violin.md", text="tuned in perfect fifths"),
            ),
        )
    ]


def _case(**overrides):
    base = dict(
        query="q",
        query_lang="en",
        type="buried",
        scenario="buried",
        expected_notes=("violin.md",),
        answer_spans=(AnswerSpan(note="violin.md", text="perfect fifths"),),
    )
    base.update(overrides)
    return WikiGoldCase(**base)


def test_validate_accepts_well_formed_case():
    notes = {"violin.md": "The violin is tuned in perfect fifths."}
    assert validate_wiki_gold([_case()], notes) == []


def test_validate_flags_missing_span_text_unknown_type_and_empty_notes():
    notes = {"violin.md": "The violin is tuned in perfect fifths."}
    cases = [
        _case(answer_spans=(AnswerSpan(note="violin.md", text="not in the article"),)),
        _case(type="mystery"),
        _case(expected_notes=()),
        _case(answer_spans=(AnswerSpan(note="missing.md", text="perfect fifths"),)),
        _case(answer_spans=()),
    ]
    errors = validate_wiki_gold(cases, notes)
    assert any("span text not found" in e for e in errors)
    assert any("unknown type" in e for e in errors)
    assert any("expected_notes is empty" in e for e in errors)
    assert any("not in corpus" in e for e in errors)
    assert any("no answer_spans" in e for e in errors)


SAMPLE = Path(__file__).resolve().parents[2] / "eval" / "wiki" / "gold.sample.json"


def test_committed_schema_sample_loads_and_has_expected_shape():
    cases = load_wiki_gold(SAMPLE)
    assert len(cases) == 2
    assert {c.type for c in cases} == {"buried", "cross_lingual"}
    assert all(c.expected_notes and c.answer_spans for c in cases)
    for case in cases:
        for span in case.answer_spans:
            assert span.note in case.expected_notes
