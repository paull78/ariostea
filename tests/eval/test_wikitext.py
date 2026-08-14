from ariostea.eval.wikitext import (
    convert_formatting,
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


def test_convert_formatting_maps_quotes_to_asterisks():
    raw = "The '''violin''' is a ''chordophone''."
    assert convert_formatting(raw) == "The **violin** is a *chordophone*."


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


def test_convert_formatting_leaves_single_apostrophes_in_prose_untouched():
    # Italian and Spanish prose is full of single apostrophes (elisions,
    # contractions); only a *pair* of quote characters means emphasis.
    raw = "L'archetto è fatto di legno, non d'acciaio. El violín no es la viola."
    assert convert_formatting(raw) == raw


def test_convert_formatting_preserves_an_apostrophe_inside_bold_text():
    assert convert_formatting("'''dell'arte''' è bello") == "**dell'arte** è bello"


def test_convert_formatting_handles_combined_bold_and_italic():
    # `'''''both'''''` (5 quotes) is real Wikipedia convention for bold+italic
    # together. The bold pass strips the outer 3 quotes from each end first,
    # leaving `''both''` for the italic pass — Markdown's own `***both***`
    # falls out without special-casing the 5-quote form.
    assert convert_formatting("'''''both'''''") == "***both***"


def test_convert_lists_reads_a_redirect_directive_as_a_numbered_item():
    # Known, accepted limitation: `#REDIRECT [[Target]]` is only valid
    # wikitext as an article's first line, and this function has no notion
    # of "first line of the article" — it reads it as an ordinary numbered
    # item. Not a real risk for this corpus: wiki_fetch.fetch_article
    # requests `redirects=1`, so a redirect page's wikitext is never handed
    # to this pipeline in the first place.
    assert convert_lists("#REDIRECT [[Target]]") == "1. REDIRECT [[Target]]"


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
