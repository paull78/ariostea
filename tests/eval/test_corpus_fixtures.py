from pathlib import Path

CORPUS = Path(__file__).resolve().parents[2] / "eval" / "corpus"


def _read(name):
    return (CORPUS / name).read_text(encoding="utf-8")


def test_accent_targets_contain_their_keyword():
    assert "città" in _read("astronomia_it.md")
    assert "montaña" in _read("ciclismo_es.md")


def test_inflection_notes_contain_only_the_base_form():
    cucito = _read("cucito_it.md")
    assert "bottone" in cucito and "bottoni" not in cucito
    alfareria = _read("alfareria_es.md")
    assert "vasija" in alfareria and "vasijas" not in alfareria
