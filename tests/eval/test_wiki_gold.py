import json

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
    ]
    errors = validate_wiki_gold(cases, notes)
    assert any("span text not found" in e for e in errors)
    assert any("unknown type" in e for e in errors)
    assert any("expected_notes is empty" in e for e in errors)
    assert any("not in corpus" in e for e in errors)
