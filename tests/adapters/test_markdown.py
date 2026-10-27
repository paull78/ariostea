"""CommonMark code fences: a `#` line inside one is code, not a heading
(vault-hardening.07)."""

from ariostea.adapters.markdown import heading_lines


def _headings(body):
    return [line for line, match in heading_lines(body) if match]


def test_headings_outside_fences_are_found():
    assert _headings("# A\ntext\n## B\n####### not one\n#nospace") == ["# A", "## B"]


def test_every_line_comes_back_in_order():
    body = "# A\n```\n# x\n```\ntail"
    assert [line for line, _ in heading_lines(body)] == body.splitlines()


def test_hash_lines_inside_backtick_fence_are_not_headings():
    assert _headings("# A\n```python\n# comment\n```\n## B") == ["# A", "## B"]


def test_hash_lines_inside_tilde_fence_are_not_headings():
    assert _headings("~~~\n# comment\n~~~\n# A") == ["# A"]


def test_indented_fence_up_to_three_spaces():
    assert _headings("   ```\n# comment\n   ```\n# A") == ["# A"]
    # Four spaces is an indented code line, not a fence: the heading counts.
    assert _headings("    ```\n# A") == ["# A"]


def test_unclosed_fence_runs_to_the_end():
    assert _headings("# A\n```\n# x\n# y") == ["# A"]


def test_longer_fence_is_not_closed_by_shorter():
    assert _headings("````\n```\n# x\n````\n# A") == ["# A"]


def test_fence_closed_by_other_char_stays_open():
    assert _headings("```\n~~~\n# x\n```\n# A") == ["# A"]


def test_closing_fence_takes_no_info_string():
    assert _headings("```\n``` python\n# x\n```\n# A") == ["# A"]


def test_backtick_fence_info_cannot_hold_backticks():
    # "``` a`b" is inline code, not a fence opening.
    assert _headings("``` a`b\n# A") == ["# A"]
