import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[2] / "eval" / "run_wiki_eval.py"
_SPEC = importlib.util.spec_from_file_location("run_wiki_eval", _PATH)
run_wiki_eval = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = run_wiki_eval
_SPEC.loader.exec_module(run_wiki_eval)


def test_missing_gold_exits_with_a_pointer_to_the_generator(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(run_wiki_eval, "GOLD", tmp_path / "absent.json")
    assert run_wiki_eval.main([]) == 2
    assert "generate_gold.py" in capsys.readouterr().err


def test_an_empty_gold_file_exits_rather_than_reporting_zeros(monkeypatch, tmp_path, capsys):
    # An empty gold set scores 0.000 across the board, which reads as a broken
    # retrieval stack rather than as a missing generation run.
    gold = tmp_path / "gold.json"
    gold.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(run_wiki_eval, "GOLD", gold)
    assert run_wiki_eval.main([]) == 2
    assert "no cases" in capsys.readouterr().err


def test_invalid_gold_is_reported_rather_than_evaluated(monkeypatch, tmp_path, capsys):
    # A span that no longer appears in its note means the corpus moved under
    # the gold. Scoring it would report stale data as a retrieval regression.
    gold = tmp_path / "gold.json"
    gold.write_text(
        '[{"query": "q", "query_lang": "en", "type": "paraphrase", "scenario": "paraphrase",'
        ' "expected_notes": ["string-instruments/violin.md"],'
        ' "answer_spans": [{"note": "string-instruments/violin.md",'
        ' "text": "a span that is not in the corpus"}]}]',
        encoding="utf-8",
    )
    monkeypatch.setattr(run_wiki_eval, "GOLD", gold)
    assert run_wiki_eval.main([]) == 3
    assert "span text not found" in capsys.readouterr().err


@pytest.mark.integration
def test_a_real_run_prints_a_report_per_channel(capsys):
    if not run_wiki_eval.GOLD.exists():
        pytest.skip("gold set not generated yet")
    assert run_wiki_eval.main([]) == 0
    out = capsys.readouterr().out
    for channel in ("DENSE", "SPARSE", "HYBRID"):
        assert channel in out
    assert "cluster" in out  # the difficulty guard table
