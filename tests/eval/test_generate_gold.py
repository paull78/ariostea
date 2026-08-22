import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "eval" / "generate_gold.py"
_SPEC = importlib.util.spec_from_file_location("generate_gold", _PATH)
generate_gold = importlib.util.module_from_spec(_SPEC)
# Required before exec_module: @dataclass resolves annotations through
# sys.modules, and the script defines its own dataclass.
sys.modules[_SPEC.name] = generate_gold
_SPEC.loader.exec_module(generate_gold)

from ariostea.eval.gold_passages import Passage  # noqa: E402
from ariostea.eval.wiki_gold import load_wiki_gold, validate_wiki_gold  # noqa: E402

NOTES = {"strings/violin.md": "# Violin\n\nThe violin is tuned in perfect fifths: G, D, A, E.\n"}
TITLES = {"strings/violin.md": "Violin"}
PASSAGE = Passage(
    note="strings/violin.md",
    heading="Tuning",
    text="The violin is tuned in perfect fifths: G, D, A, E.",
    offset=10,
    note_chars=9000,
    rare_terms=("ponticello",),
    rare_score=40.0,
)

GOOD_GENERATION = (
    '{"query": "which four notes does the instrument sound open", '
    '"answer_span": "tuned in perfect fifths: G, D, A, E"}'
)
APPROVAL = '{"answers": true, "unambiguous": true, "title_only": false, "reason": "ok"}'


class FakeChat:
    def __init__(self, *responses: str) -> None:
        self._responses = list(responses)
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self._responses.pop(0) if self._responses else "not json"


def test_a_good_candidate_becomes_a_gold_case():
    cases, rejections = generate_gold.generate_and_gate(
        FakeChat(GOOD_GENERATION), FakeChat(APPROVAL), [("paraphrase", PASSAGE)], NOTES, TITLES
    )
    assert rejections == []
    assert len(cases) == 1
    assert cases[0].expected_notes == ("strings/violin.md",)
    assert cases[0].answer_spans[0].text == "tuned in perfect fifths: G, D, A, E"
    assert cases[0].scenario == "paraphrase"


def test_a_cross_lingual_case_gets_the_arrow_scenario():
    generation = (
        '{"query": "quali note produce lo strumento", '
        '"answer_span": "tuned in perfect fifths: G, D, A, E"}'
    )
    cases, rejections = generate_gold.generate_and_gate(
        FakeChat(generation), FakeChat(APPROVAL), [("cross_lingual", PASSAGE)], NOTES, TITLES
    )
    assert rejections == []
    assert cases[0].query_lang in {"it", "es"}
    assert cases[0].scenario == f"en→{cases[0].query_lang}"


def test_cross_lingual_cases_alternate_between_the_two_languages():
    generations = [
        '{"query": "quali note produce lo strumento", '
        '"answer_span": "tuned in perfect fifths: G, D, A, E"}',
        '{"query": "cuales notas produce el instrumento", '
        '"answer_span": "tuned in perfect fifths: G, D, A, E"}',
    ]
    cases, _ = generate_gold.generate_and_gate(
        FakeChat(*generations),
        FakeChat(APPROVAL, APPROVAL),
        [("cross_lingual", PASSAGE), ("cross_lingual", PASSAGE)],
        NOTES,
        TITLES,
    )
    assert [c.query_lang for c in cases] == ["it", "es"]


def test_an_unparseable_generation_is_recorded_as_a_generate_rejection():
    cases, rejections = generate_gold.generate_and_gate(
        FakeChat("sorry"), FakeChat(APPROVAL), [("paraphrase", PASSAGE)], NOTES, TITLES
    )
    assert cases == []
    assert rejections[0].stage == "generate"
    assert rejections[0].type == "paraphrase"


def test_a_fabricated_span_is_recorded_as_an_automatic_rejection():
    bad = '{"query": "what tuning is used here", "answer_span": "tuned in perfect fourths"}'
    cases, rejections = generate_gold.generate_and_gate(
        FakeChat(bad), FakeChat(APPROVAL), [("paraphrase", PASSAGE)], NOTES, TITLES
    )
    assert cases == []
    assert rejections[0].stage == "automatic" and "verbatim" in rejections[0].reason


def test_a_judge_veto_is_recorded_as_an_adversarial_rejection():
    veto = '{"answers": false, "unambiguous": true, "title_only": false, "reason": "no"}'
    cases, rejections = generate_gold.generate_and_gate(
        FakeChat(GOOD_GENERATION), FakeChat(veto), [("paraphrase", PASSAGE)], NOTES, TITLES
    )
    assert cases == []
    assert rejections[0].stage == "adversarial"


def test_the_judge_is_not_called_when_stage_one_already_rejected():
    # Stage 2 costs a model call; spending it on a span that is not even in
    # the note is pure waste over a ~150-candidate run.
    bad = '{"query": "what tuning is used here", "answer_span": "not in the passage at all"}'
    judge = FakeChat(APPROVAL)
    generate_gold.generate_and_gate(FakeChat(bad), judge, [("paraphrase", PASSAGE)], NOTES, TITLES)
    assert judge.calls == 0


def test_one_failing_candidate_does_not_stop_the_run():
    cases, rejections = generate_gold.generate_and_gate(
        FakeChat("sorry", GOOD_GENERATION),
        FakeChat(APPROVAL),
        [("paraphrase", PASSAGE), ("buried", PASSAGE)],
        NOTES,
        TITLES,
    )
    assert len(cases) == 1 and len(rejections) == 1


def test_a_chat_error_is_recorded_rather_than_raised():
    from ariostea.adapters.chat.openai_compat import ChatError

    class Broken:
        def complete(self, system, user):
            raise ChatError("connection refused")

    cases, rejections = generate_gold.generate_and_gate(
        Broken(), FakeChat(APPROVAL), [("paraphrase", PASSAGE)], NOTES, TITLES
    )
    assert cases == []
    assert rejections[0].stage == "generate" and "connection refused" in rejections[0].reason


def test_written_gold_reloads_and_validates(tmp_path):
    cases, _ = generate_gold.generate_and_gate(
        FakeChat(GOOD_GENERATION), FakeChat(APPROVAL), [("paraphrase", PASSAGE)], NOTES, TITLES
    )
    path = tmp_path / "gold.json"
    generate_gold.write_gold(path, cases)
    reloaded = load_wiki_gold(path)
    assert reloaded == cases
    assert validate_wiki_gold(reloaded, NOTES) == []


def test_written_gold_keeps_non_ascii_readable(tmp_path):
    cases, _ = generate_gold.generate_and_gate(
        FakeChat(
            '{"query": "perché è accordato così", '
            '"answer_span": "tuned in perfect fifths: G, D, A, E"}'
        ),
        FakeChat(APPROVAL),
        [("cross_lingual", PASSAGE)],
        NOTES,
        TITLES,
    )
    path = tmp_path / "gold.json"
    generate_gold.write_gold(path, cases)
    assert "perché" in path.read_text(encoding="utf-8")


def test_review_markdown_renders_a_reviewable_checklist():
    cases, _ = generate_gold.generate_and_gate(
        FakeChat(GOOD_GENERATION), FakeChat(APPROVAL), [("paraphrase", PASSAGE)], NOTES, TITLES
    )
    text = generate_gold.review_markdown(cases, sample_size=1)
    assert "which four notes" in text
    assert "strings/violin.md" in text
    assert "- [ ]" in text  # a checkbox the reviewer actually ticks


def test_review_markdown_handles_an_empty_gold_set():
    assert "no cases" in generate_gold.review_markdown([], sample_size=20).lower()


def test_rejection_summary_counts_by_stage_and_reason():
    summary = generate_gold.rejection_summary(
        [
            generate_gold.Rejection("automatic", "span is not verbatim", "q", "n", "s", "para"),
            generate_gold.Rejection("automatic", "span is not verbatim", "q", "n", "s", "buried"),
            generate_gold.Rejection("adversarial", "judge: ambiguous (x)", "q", "n", "s", "buried"),
        ]
    )
    assert "automatic" in summary and "adversarial" in summary
    assert "2" in summary


def test_rejection_summary_groups_reasons_by_cause_not_phrasing():
    # Judge reasons carry the model's free text in parentheses; two rejections
    # for the same cause must not read as two different causes.
    summary = generate_gold.rejection_summary(
        [
            generate_gold.Rejection(
                "adversarial", "judge: ambiguous (two readings)", "", "", "", "p"
            ),
            generate_gold.Rejection("adversarial", "judge: ambiguous (unclear)", "", "", "", "p"),
        ]
    )
    assert summary.count("judge: ambiguous") == 1


def test_rejection_summary_of_nothing_is_empty():
    assert generate_gold.rejection_summary([]) == ""


def test_shortfall_is_reported_not_swallowed(capsys):
    generate_gold.report_shortfall({"paraphrase": 40}, [("paraphrase", PASSAGE)])
    assert "paraphrase" in capsys.readouterr().err


def test_no_shortfall_is_reported_when_the_budget_is_met(capsys):
    generate_gold.report_shortfall({"paraphrase": 1}, [("paraphrase", PASSAGE)])
    assert capsys.readouterr().err == ""


def test_generation_finishes_for_every_passage_before_the_judge_is_called():
    # Both models cannot sit in memory at once on a 48GB machine, so
    # alternating per candidate would make LM Studio swap models ~300 times
    # over a 150-passage run. Batching the stages costs one swap.
    order: list[str] = []

    class Recording:
        def __init__(self, label, response):
            self.label, self.response = label, response

        def complete(self, system, user):
            order.append(self.label)
            return self.response

    generate_gold.generate_and_gate(
        Recording("gen", GOOD_GENERATION),
        Recording("judge", APPROVAL),
        [("paraphrase", PASSAGE), ("buried", PASSAGE)],
        NOTES,
        TITLES,
    )
    assert order == ["gen", "gen", "judge", "judge"]


def test_batching_still_skips_the_judge_for_stage_one_rejects():
    bad = '{"query": "what tuning is used here", "answer_span": "not in the passage at all"}'
    judge = FakeChat(APPROVAL)
    cases, rejections = generate_gold.generate_and_gate(
        FakeChat(bad, GOOD_GENERATION),
        judge,
        [("paraphrase", PASSAGE), ("buried", PASSAGE)],
        NOTES,
        TITLES,
    )
    assert judge.calls == 1  # only the surviving candidate reached stage 2
    assert len(cases) == 1 and len(rejections) == 1
