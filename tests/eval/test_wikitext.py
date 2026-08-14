from ariostea.eval.wikitext import (
    convert_emphasis,
    convert_headings,
    convert_lists,
    drop_sections,
    normalize_blank_lines,
    strip_comments,
    strip_html_containers,
    strip_html_tags,
    strip_media_links,
    strip_refs,
    strip_tables,
    strip_templates,
)


def test_strip_comments_and_refs():
    raw = "A<!-- hidden -->B<ref name=x>Smith 2001</ref>C<ref name=y />D"
    assert strip_refs(strip_comments(raw)) == "ABCD"


def test_strip_refs_removes_the_references_placeholder():
    assert strip_refs("body\n<references/>\n") == "body\n\n"


def test_strip_refs_removes_references_container_with_nested_ref_definitions():
    raw = "Prose.\n<references>\n<ref name=a>Foo</ref>\n</references>\nMore."
    assert strip_refs(raw) == "Prose.\n\nMore."


def test_strip_refs_unclosed_ref_does_not_eat_prose_up_to_the_next_ref():
    # A missing `</ref>` must not turn the lazy body into a scanner that runs
    # to the *next* ref's closer, deleting everything in between. The
    # unclosed opener leaks its own literal `<ref>` instead.
    raw = "A<ref>unclosed. Real prose paragraph here.<ref>cite</ref>C"
    assert strip_refs(raw) == "A<ref>unclosed. Real prose paragraph here.C"


def test_strip_refs_removes_self_closing_ref_placed_before_a_paired_one():
    # Pins the intra-function order: self-closing refs must be substituted
    # before paired refs, or the paired pattern's opening-tag match can
    # swallow a self-closing ref that appears earlier in the text (its `/`
    # just looks like more attribute soup to `[^>]*`).
    raw = "<ref name=y />A<ref name=x>Smith 2001</ref>"
    assert strip_refs(raw) == "A"


def test_strip_templates_handles_nesting():
    assert strip_templates("a {{convert|4|{{frac|1|2}} ft}} b") == "a  b"


def test_strip_tables_removes_the_whole_table():
    assert strip_tables("x\n{| class=wikitable\n|-\n| cell\n|}\ny") == "x\n\ny"


def test_strip_tables_handles_nesting():
    raw = "x\n{| outer\n| {| inner\n| cell\n|}\n|}\ny"
    assert strip_tables(raw) == "x\n\ny"


def test_strip_media_links_removes_captions_including_nested_links():
    assert strip_media_links("[[File:V.jpg|thumb|A [[violin]] on a table]]Text") == "Text"
    assert strip_media_links("[[Immagine:V.jpg|thumb|foto]]Testo") == "Testo"


def test_strip_media_links_leaves_ordinary_links_alone():
    assert strip_media_links("a [[violin]] b") == "a [[violin]] b"


def test_strip_media_links_leaves_media_audio_links_alone():
    # [[Media:...]] is an inline prose link ("listen to this clip"), not an
    # embed — Task 3's link rewriter flattens it to its label like any other
    # link. Stripping it whole here would eat the anchor text.
    assert strip_media_links("[[Media:song.ogg|listen]] here") == "[[Media:song.ogg|listen]] here"


def test_strip_media_links_handles_spanish_imagen_alias():
    assert strip_media_links("[[Imagen:foo.jpg|thumb|pie]]Texto") == "Texto"


def test_strip_html_containers_removes_gallery_with_content():
    assert strip_html_containers("<gallery>\nFile:x.jpg\n</gallery>b") == "b"


def test_strip_html_containers_removes_math_and_code_blocks():
    raw = '<math>x^2</math> and <syntaxhighlight lang="python">print(1)</syntaxhighlight>b'
    assert strip_html_containers(raw) == " and b"


def test_strip_html_tags_removes_inline_tags():
    assert strip_html_tags("<small>a</small>b") == "ab"


def test_strip_html_tags_catches_tags_not_in_a_hardcoded_list():
    raw = "<b>bold</b> <table><tr><td>x</td></tr></table> y"
    assert strip_html_tags(raw) == "bold x y"


# --- unbalanced / unterminated constructs: fix 1 -----------------------------
#
# A missing closer must never silently truncate an article. The fixed
# behavior is to leak the unclosed region verbatim (loud and greppable)
# instead of discarding it (invisible and indistinguishable from a real
# retrieval failure downstream).


def test_strip_templates_leaks_unterminated_template_verbatim():
    raw = "intro {{broken\n\n== Section ==\nreal prose"
    assert strip_templates(raw) == raw


def test_strip_tables_leaks_unterminated_table_verbatim():
    raw = "x\n{| unterminated\n| cell\ny"
    assert strip_tables(raw) == raw


def test_strip_media_links_leaks_unterminated_media_link_verbatim():
    raw = "A [[File:ok.jpg|thumb|fine]] B [[File:bad.jpg|thumb|broken"
    assert strip_media_links(raw) == "A  B [[File:bad.jpg|thumb|broken"


# --- ordinary prose is left alone --------------------------------------------


def test_ordinary_text_with_brace_pipe_angle_characters_is_left_alone():
    text = "Cost: {estimate} | rate < 5 and value > 10"
    assert strip_templates(text) == text
    assert strip_tables(text) == text
    assert strip_html_tags(text) == text


def test_strip_html_tags_catch_all_does_not_cross_a_newline():
    # A spaced inequality on one line is a trivial case for the catch-all —
    # it never has to look past the next character. An *unspaced* `<`/`>`
    # separated by a real paragraph break is the actual hazard: `[^>]*` under
    # no line limit would swallow everything up to the first `>` anywhere
    # later in the article. Bounding the pattern to `[^>\n]*` turns that into
    # "no match on this line" instead of "delete a whole paragraph".
    text = "range a<b.\n\nA whole paragraph of real prose.\n\nThen x>y."
    assert strip_html_tags(text) == text


# --- pipeline order interaction: fix 4 ---------------------------------------


def test_comment_stripped_before_template_scan_avoids_brace_poisoning():
    raw = "Intro <!-- {{maybe --> real prose"
    assert strip_templates(strip_comments(raw)) == "Intro  real prose"


def test_template_scan_without_comment_stripping_first_leaks_the_comment_brace():
    # Demonstrates *why* the pipeline order matters: skip the comments step
    # and the comment's own "{{" poisons the scanner, leaking everything from
    # that point on (fix 1) instead of just the comment.
    raw = "Intro <!-- {{maybe --> real prose"
    assert strip_templates(raw) == raw


# --- realistic composite fixture ---------------------------------------------


def test_pipeline_strips_chrome_from_an_article_lede():
    raw = (
        "The violin is a string instrument.<!-- verify --> "
        'It has four strings.<ref name="grove">Grove, Dictionary of Music</ref> '
        "The body uses [[spruce]] and [[maple]].\n\n"
        "{{Infobox instrument|name=Violin|range={{range|G3|E7}}}}\n\n"
        "[[File:Violin.png|thumb|A violin with [[chin rest]] fitted.]]\n\n"
        "== Construction ==\n"
        '{| class="wikitable"\n|-\n! Part !! Material\n|-\n| Belly || Spruce\n|}\n'
        "<gallery>\nFile:Violin front.jpg|Front\n</gallery>\n"
        "<small>See also the [[viola]].</small>\n"
        "<references/>\n"
    )
    text = raw
    for step in (
        strip_comments,
        strip_refs,
        strip_html_containers,
        strip_templates,
        strip_tables,
        strip_media_links,
        strip_html_tags,
    ):
        text = step(text)

    # chrome is gone
    assert "verify" not in text
    assert "Grove" not in text
    assert "<ref" not in text
    assert "{{" not in text and "}}" not in text
    assert "{|" not in text and "|}" not in text
    assert "[[File:" not in text
    assert "<gallery" not in text and "Front" not in text
    assert "<small>" not in text and "</small>" not in text
    assert "chin rest" not in text

    # real prose and ordinary links survive
    assert "The violin is a string instrument." in text
    assert "[[spruce]]" in text
    assert "[[maple]]" in text
    assert "[[viola]]" in text
    assert "Construction" in text


def test_convert_headings_maps_equals_depth_to_hashes():
    assert convert_headings("== Construction ==\n=== Body ===") == "## Construction\n### Body"


def test_convert_lists_handles_bullets_numbers_and_definitions():
    raw = "* one\n** two\n# first\n## second\n: indented"
    assert convert_lists(raw) == "- one\n  - two\n1. first\n  1. second\nindented"


def test_convert_emphasis_maps_quotes_to_asterisks():
    raw = "The '''violin''' is a ''chordophone''."
    assert convert_emphasis(raw) == "The **violin** is a *chordophone*."


def test_drop_sections_removes_the_section_and_its_subsections():
    md = "## Construction\nbody\n\n## References\n- r1\n### Notes\nx\n\n## Playing\nmusic"
    assert drop_sections(md) == "## Construction\nbody\n\n## Playing\nmusic"


def test_drop_sections_covers_italian_and_spanish_boilerplate():
    assert drop_sections("## Storia\nc\n## Voci correlate\nx") == "## Storia\nc"
    assert drop_sections("## Historia\nc\n## Enlaces externos\nx") == "## Historia\nc"


def test_normalize_blank_lines_collapses_runs_and_trailing_space():
    assert normalize_blank_lines("a  \n\n\n\nb\n") == "a\n\nb"


# --- real-input hazards, probed against real Wikipedia conventions ----------


def test_convert_headings_uses_the_shorter_side_when_equals_counts_mismatch():
    # A genuine (if rare) editing typo: 3 leading `=`, 2 trailing. `\1`
    # backtracks `(={2,6})` to the shorter run (2), and the surplus `=` on
    # the longer side falls into the title as literal text — the same
    # level-and-leftover result MediaWiki's own heading parser produces.
    assert convert_headings("=== Title ==\nbody") == "## = Title\nbody"


def test_convert_headings_caps_at_level_six_for_a_deeper_marker_run():
    assert convert_headings("======= Deep =======") == "###### = Deep ="


def test_convert_emphasis_leaves_single_apostrophes_in_prose_untouched():
    # Italian and Spanish prose is full of single apostrophes (elisions,
    # contractions); only a *pair* of quote characters means emphasis.
    raw = "L'archetto è fatto di legno, non d'acciaio. El violín no es la viola."
    assert convert_emphasis(raw) == raw


def test_convert_emphasis_preserves_an_apostrophe_inside_bold_text():
    assert convert_emphasis("'''dell'arte''' è bello") == "**dell'arte** è bello"


def test_convert_emphasis_handles_combined_bold_and_italic():
    # `'''''both'''''` (5 quotes) is real Wikipedia convention for bold+italic
    # together. The bold pass strips the outer 3 quotes from each end first,
    # leaving `''both''` for the italic pass — Markdown's own `***both***`
    # falls out without special-casing the 5-quote form.
    assert convert_emphasis("'''''both'''''") == "***both***"


def test_convert_emphasis_leaves_an_unmatched_bold_marker_in_place():
    # Same invariant as the strip functions above: an unmatched opener is
    # left untouched rather than guessed at.
    assert convert_emphasis("a '''unclosed bold here") == "a '''unclosed bold here"


# --- list-prefix ordering: item 1 --------------------------------------------
#
# `:#` and `:*` are real wikitext (an indented numbered/bulleted item, common
# in bibliography and notes sections). `_DEF_LINE` must run *first* inside
# `convert_lists`: if it ran last, stripping the leading `:` would re-expose
# a bare `#`/`*` line for `_NUMBERED`/`_BULLET` to have already skipped over,
# leaking a Markdown heading/bullet marker downstream instead of converting
# it. See `convert_lists`'s docstring.


def test_convert_lists_does_not_leak_a_bare_hash_from_an_indented_numbered_item():
    assert convert_lists(":# Smith 2001") == "1. Smith 2001"


def test_convert_lists_output_does_not_get_misread_as_a_heading_by_drop_sections():
    # The defect at the point it actually destroys prose: `:# See also`
    # inside a *kept* section, if list-converted wrong, becomes a bare
    # `# See also` — a real Markdown H1 whose title `drop_sections` would
    # skip to end-of-article on, silently deleting everything after it.
    md = "== Construction ==\nbody\n:# See also\nmore body\n== Playing ==\nmusic"
    converted = convert_headings(convert_lists(md))
    assert converted == "## Construction\nbody\n1. See also\nmore body\n## Playing\nmusic"
    assert drop_sections(converted) == converted


def test_convert_lists_output_does_not_undrop_a_boilerplate_section():
    # `:# Smith 2001` is a real bibliography line inside a References
    # section. If `_DEF_LINE` ran last, the surviving `# Smith 2001` would
    # be a level-1 heading *inside* the dropped level-2 References section —
    # shallower than it, which would end the skip early and un-drop the rest
    # of the section.
    md = "== References ==\n:# Smith 2001\nmore citations"
    converted = convert_headings(convert_lists(md))
    assert drop_sections(converted) == ""


# --- lists-before-emphasis ordering: item 2 ----------------------------------


def test_lists_before_emphasis_keeps_the_lede_intact():
    # Every article's lede reads `'''Title''' is a ...`. This is the order
    # Task 3's composition actually uses.
    raw = "'''Violin''' is a string instrument."
    assert convert_emphasis(convert_lists(raw)) == "**Violin** is a string instrument."


def test_emphasis_before_lists_corrupts_the_lede_into_a_bullet():
    # Demonstrates *why* the order matters: run emphasis first and `'''`
    # becomes `**`, which `_BULLET`'s `^(\*+)` then reads as a two-deep
    # bullet prefix, eating the emphasis and the article's opening sentence.
    raw = "'''Violin''' is a string instrument."
    assert convert_lists(convert_emphasis(raw)) == "  - Violin** is a string instrument."


def test_drop_sections_drops_to_end_of_article_when_the_section_is_last():
    # The blank separator line belongs to the *kept* section, not the
    # dropped one, so it survives even though everything after it doesn't.
    md = "## Construction\nbody\n\n## References\n- r1\n- r2"
    assert drop_sections(md) == "## Construction\nbody\n"


def test_drop_sections_matches_titles_that_differ_only_by_case():
    assert drop_sections("## Storia\nc\n## SEE ALSO\nx") == "## Storia\nc"


def test_drop_sections_matches_an_accented_title_regardless_of_case():
    assert drop_sections("## Historia\nc\n## VÉASE TAMBIÉN\nx") == "## Historia\nc"


def test_drop_sections_drops_a_boilerplate_subsection_nested_in_a_kept_section():
    md = "## Construction\nbody\n### See also\nx\n## Playing\nmusic"
    assert drop_sections(md) == "## Construction\nbody\n## Playing\nmusic"


def test_drop_sections_matches_a_title_wrapped_in_emphasis_markers():
    # Task 3 converts emphasis after headings, so a wikitext heading written
    # as `== '''References''' ==` arrives here as `## **References**`. The
    # wrapping `**`/`_` must not hide the title from the boilerplate lookup.
    assert drop_sections("## Storia\nc\n## **References**\nx") == "## Storia\nc"


def test_drop_sections_ends_the_skip_at_a_shallower_heading_not_just_equal_depth():
    # The `level <= skip_level` branch is only ever exercised at *equal*
    # depth by the tests above; a strictly shallower heading must end the
    # skip too, even though its own title isn't boilerplate.
    md = "## References\nstuff\n# Playing\nmusic"
    assert drop_sections(md) == "# Playing\nmusic"


def test_drop_sections_accepts_a_custom_titles_set():
    # `titles` compares against a stripped, lower-cased heading — a caller
    # supplying its own set has to lower-case its own titles to match.
    assert drop_sections("## Keep\nx\n## Drop\ny", titles=frozenset({"drop"})) == "## Keep\nx"


def test_structure_conversion_leaves_a_later_paragraph_intact():
    # Every hazard test above is one or two lines of input. This fixture
    # checks that converting one paragraph's structure (list, heading,
    # emphasis) doesn't bleed into the next one — the recurring failure mode
    # Task 1 guards against with test_strip_html_tags_catch_all_does_not_cross_a_newline.
    raw = (
        "== Construction ==\n"
        "* four strings\n"
        "* tuned in fifths\n"
        "\n"
        "The '''violin''' body is carved from spruce and maple.\n"
    )
    text = raw
    for step in (convert_lists, convert_headings, convert_emphasis):
        text = step(text)
    assert text == (
        "## Construction\n"
        "- four strings\n"
        "- tuned in fifths\n"
        "\n"
        "The **violin** body is carved from spruce and maple.\n"
    )
