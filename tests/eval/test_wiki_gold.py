import json

from ariostea.eval.wiki_gold import AnswerSpan, WikiGoldCase, load_wiki_gold


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
