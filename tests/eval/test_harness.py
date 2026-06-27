from ariostea.eval.harness import GoldCase, load_gold


def test_load_gold_parses_cases(tmp_path):
    gold = tmp_path / "gold.json"
    # → is the "→" arrow; written escaped to keep the test ASCII-safe.
    gold.write_text(
        '[{"query": "dice game", "query_lang": "en", '
        '"expected": ["dadi_it.md"], "direction": "en\\u2192it"}]',
        encoding="utf-8",
    )

    cases = load_gold(gold)

    assert cases == [
        GoldCase(
            query="dice game",
            query_lang="en",
            expected=("dadi_it.md",),
            direction="en→it",
        )
    ]
