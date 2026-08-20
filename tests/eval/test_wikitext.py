from collections import Counter

from ariostea.eval.wikitext import (
    convert_emphasis,
    convert_headings,
    convert_links,
    convert_lists,
    decode_entities,
    drop_sections,
    expand_templates,
    normalize_blank_lines,
    restore_apostrophe_placeholders,
    strip_comments,
    strip_empty_emphasis,
    strip_external_links,
    strip_html_containers,
    strip_html_tags,
    strip_media_links,
    strip_refs,
    strip_tables,
    strip_templates,
    tidy_punctuation,
    wikitext_to_markdown,
)

TARGETS = {
    "violin": "violin",
    "double bass": "double-bass",
    "string instrument": "string-instrument",
}


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


# --- empty-emphasis cleanup: a template that was an emphasis span's entire
# content -------------------------------------------------------------------
#
# Found in the trial build against real Wikipedia data (Task 8), not
# invented: the English "Luthier" article (rev 1365251608) opens
# `''{{Wikt-lang|fr|luthier}}'' is originally ... from ''luth''`. Once
# strip_templates removes the template, the first span is an empty italic
# pair (`''''`) sitting right in front of a second, unrelated, real italic
# span later in the same sentence.


def test_strip_empty_emphasis_removes_an_empty_italic_pair_left_by_a_stripped_template():
    assert strip_empty_emphasis("word '''' is empty") == "word  is empty"


def test_strip_empty_emphasis_removes_an_empty_bold_pair_left_by_a_stripped_template():
    assert strip_empty_emphasis("word '''''' is empty") == "word  is empty"


def test_strip_empty_emphasis_leaves_real_emphasis_alone():
    raw = "The '''violin''' is a ''chordophone''."
    assert strip_empty_emphasis(raw) == raw


def test_convert_emphasis_without_cleanup_corrupts_a_later_unrelated_italic_span():
    # Demonstrates the defect itself: convert_emphasis's ITALIC regex, with
    # no notion of "empty", scans past the leftover `''''` hunting for a
    # legitimate close and instead consumes into the next real ''luth''
    # span -- corrupting the boundary and leaking two literal apostrophes.
    raw = (
        "The word '''' is originally French and comes from ''luth'', the French word for \"lute\"."
    )
    assert convert_emphasis(raw) == (
        "The word *'' is originally French and comes from *luth'', the French word for \"lute\"."
    )


def test_strip_empty_emphasis_before_convert_emphasis_keeps_the_later_span_intact():
    raw = (
        "The word '''' is originally French and comes from ''luth'', the French word for \"lute\"."
    )
    cleaned = convert_emphasis(strip_empty_emphasis(raw))
    assert cleaned == (
        'The word  is originally French and comes from *luth*, the French word for "lute".'
    )


def test_wikitext_to_markdown_does_not_corrupt_emphasis_when_a_template_was_a_spans_entire_content():
    # End-to-end pin of the real Luthier construct through the full
    # pipeline, not just the two functions in isolation.
    raw = (
        "''{{Wikt-lang|fr|luthier}}'' is originally [[French language|French]] "
        "and comes from ''luth'', the French word for \"[[lute]]\"."
    )
    body = wikitext_to_markdown(raw, title="Luthier", targets={"lute": "lute"})
    assert "*'" not in body
    assert "*luth*" in body
    assert "[[lute]]" in body


# --- display-template expansion: convert/frac/lang/circa/music/nowrap -------
#
# Every shape below is a real invocation confirmed against six fetched
# articles (Violin rev 1366434192, Viola rev 1364838851, Cello rev
# 1360203898, Double bass rev 1365602377, Classical guitar rev 1368953989,
# Mandolin rev 1368537001) before the allowlist in wikitext.py was written —
# not invented cases. See `_DISPLAY_TEMPLATES`'s comment for which shapes
# came from that sample versus the original task spec.


def test_expand_templates_convert_drops_the_output_unit_and_precision():
    # {{convert|356|mm|in|1|abbr=on}}, real (Violin): value + first unit
    # only; target unit, precision digit, and every named arg are dropped.
    assert expand_templates("{{convert|356|mm|in|1|abbr=on}}") == "356 mm"


def test_expand_templates_convert_handles_the_to_range_form():
    assert expand_templates("{{convert|4|to|6|ft}}") == "4 to 6 ft"


def test_expand_templates_convert_handles_the_x_and_en_dash_range_forms():
    assert expand_templates("{{convert|4|x|6|ft}}") == "4 x 6 ft"
    # MediaWiki sets a dash range tight and spaces a worded one; verified
    # against action=expandtemplates.
    assert expand_templates("{{convert|484|–|578|mm|in|abbr=on}}") == "484–578 mm"


def test_expand_templates_convert_handles_the_and_joiner():
    # {{convert|20|and|22|in}}, real (Viola, Mandolin). Not in the original
    # task spec -- added because dropping it silently would render "20"
    # with the "22 in" half of the measurement gone, not just an
    # unconverted joiner.
    assert expand_templates("{{convert|20|and|22|in}}") == "20 and 22 in"


def test_expand_templates_convert_normalizes_the_to_hyphen_variant():
    # {{convert|60|to(-)|75|cm|in}}, real (Double bass). MediaWiki's own
    # notation for "join with a hyphen instead of the word 'to'";
    # normalized to the same "to" text as the plain form rather than
    # reproduced literally.
    assert expand_templates("{{convert|60|to(-)|75|cm|in}}") == "60 to 75 cm"


def test_expand_templates_convert_generically_strips_the_hyphen_suffix_from_any_joiner():
    # {{convert|38|and(-)|46|cm|in|abbr=on|disp=sqbr}}, real (Viola,
    # viola.md's committed trial build). The first cut of this fix only
    # special-cased "to(-)" as its own dict key, which is exactly why this
    # one slipped through: "and(-)" wasn't a key either. Fixed generically
    # (see _convert_joiner) so any "<word>(-)" variant resolves without
    # needing its own entry.
    assert expand_templates("{{convert|38|and(-)|46|cm|in|abbr=on|disp=sqbr}}") == "38 and 46 cm"


def test_expand_templates_convert_expands_the_mixed_number_value_syntax():
    # {{convert|13+7/8|in}}, real (Cello, Mandolin) -- convert's own
    # compact mixed-number notation, distinct from a nested {{frac}}.
    assert expand_templates("{{convert|13+7/8|in}}") == "13 7/8 in"


def test_expand_templates_frac_handles_bare_whole_and_mixed_forms():
    assert expand_templates("{{frac|1|2}}") == "1/2"
    assert expand_templates("{{frac|3|1|2}}") == "3 1/2"
    assert expand_templates("{{frac}}") == ""


def test_expand_templates_convert_and_frac_nest_innermost_first():
    # The construct this fix was specifically required to handle: frac
    # resolves to plain text on the first pass, which then makes the outer
    # convert brace-free (innermost) on the second pass.
    assert expand_templates("{{convert|4|{{frac|1|2}} ft}}") == "4 1/2 ft"


def test_expand_templates_lang_and_wikt_lang_emit_the_last_positional_arg():
    # {{lang|it|violino}} and {{Wikt-lang|fr|luthier}}, both real -- the
    # latter is the exact Luthier (rev 1365251608) construct that motivated
    # the strip_empty_emphasis fix; wikt-lang now expands at the source
    # instead of relying solely on that cleanup stage downstream.
    assert expand_templates("{{lang|it|violino}}") == "violino"
    assert expand_templates("{{Wikt-lang|fr|luthier}}") == "luthier"


def test_expand_templates_lang_preserves_a_nested_wikilink():
    # {{lang|it|[[Viola da gamba]]}}, real (Violin). The wikilink survives
    # as ordinary wikitext for convert_links to resolve later in the
    # pipeline -- expand_templates has no notion of link syntax at all.
    assert expand_templates("{{lang|it|[[Viola da gamba]]}}") == "[[Viola da gamba]]"


def test_expand_templates_circa_bare_and_with_year():
    assert expand_templates("{{circa}}") == "c."
    assert expand_templates("{{circa|1700}}") == "c. 1700"


def test_expand_templates_music_maps_both_shorthand_and_word_forms():
    # {{music|#}}/{{music|b}} and {{music|flat}}/{{music|sharp}} both
    # appear in the sample for the same symbols -- real editors mix both
    # conventions in the same article.
    assert expand_templates("{{music|flat}}") == "♭"
    assert expand_templates("{{music|b}}") == "♭"
    assert expand_templates("{{music|sharp}}") == "♯"
    assert expand_templates("{{music|#}}") == "♯"
    assert expand_templates("{{music|natural}}") == "♮"


def test_expand_templates_music_time_signature():
    # {{music|time|3|4}}, real (Cello, Mandolin) -- not in the original
    # task spec's mapping (which only covered accidentals); dropping it
    # would delete a fact like "written in 3/4 time" the same way an
    # unexpanded {{convert}} deletes a measurement.
    assert expand_templates("{{music|time|3|4}}") == "3/4"


def test_expand_templates_music_drops_an_unrecognized_argument():
    assert expand_templates("{{music|breve}}") == ""


def test_expand_templates_nowrap_emits_its_contents():
    assert expand_templates("{{nowrap|100 mm}}") == "100 mm"


def test_expand_templates_nobr_is_a_real_nowrap_alias():
    # {{nobr}}, real (11x across the 18-article trial corpus, found via the
    # dropped-template report -- Wikipedia's own Nobr template redirects to
    # Nowrap, so this module treats them the same way.
    assert expand_templates("{{nobr|100 mm}}") == "100 mm"


def test_expand_templates_blockquote_quotes_the_text_with_attribution():
    # {{blockquote|text|author|source}}, real (Classical guitar, rev
    # 1368953989) -- found via the dropped-template report after an entire
    # multi-sentence quoted interview silently vanished from
    # classical-guitar.md in the committed trial build.
    raw = "{{blockquote|Do not understand me wrong.|Bernard Hebb|Interview}}"
    assert expand_templates(raw) == '"Do not understand me wrong." — Bernard Hebb, Interview'


def test_expand_templates_blockquote_with_no_attribution():
    assert expand_templates("{{blockquote|Just the quote.}}") == '"Just the quote."'


def test_expand_templates_blockquote_handles_the_named_arg_form():
    # Wikipedia's real Template:Blockquote documents text=/author=/source=
    # as the canonical form; the one construct actually seen in this corpus
    # uses positional args instead (see the test above), so both are
    # supported.
    raw = "{{blockquote|text=Named form.|author=Someone}}"
    assert expand_templates(raw) == '"Named form." — Someone'


def test_wikitext_to_markdown_no_longer_drops_the_real_blockquote_paragraph():
    # End-to-end regression for the classical-guitar.md defect: a whole
    # paragraph of unique prose must survive, not just a value or symbol.
    raw = (
        "Julian Bream is admired. "
        "{{blockquote|The last guitarist to follow in Segovia's footsteps "
        "was Julian Bream.|Bernard Hebb|Interview}} "
        "He remains influential today."
    )
    body = wikitext_to_markdown(raw, title="Classical guitar", targets={})
    assert "The last guitarist to follow in Segovia's footsteps" in body
    assert "Bernard Hebb" in body
    assert "{{" not in body


# --- cvt/langx/siglo/floruit/formatnum/apostrophe: second dropped-report
# review -----------------------------------------------------------------
#
# High-confidence, high-frequency names only, filtered from the same
# dropped-template report: aliases of handlers that already exist, or a
# trivial one-token expansion whose correct output could be stated (or, for
# siglo, confirmed against the live template) without guessing. Everything
# else the report turned up stays dropped and visible in it -- see the
# module comment above `_split_template_args` for the full list and why.


def test_expand_templates_cvt_is_a_real_convert_alias():
    # {{cvt|1.5|and(-)|2|mm|2}}, real (Lute, rev 1361649316) -- shares
    # _expand_convert, including its and(-) generic joiner handling.
    assert expand_templates("{{cvt|26.25|in|mm}}") == "26.25 in"
    assert expand_templates("{{cvt|1.5|and(-)|2|mm|2}}") == "1.5 and 2 mm"


def test_expand_templates_langx_is_a_real_lang_alias():
    # {{langx|it|mandolino}} and {{langx|de|link=no|Bratsche}}, both real
    # (Viola rev 1364838851, Mandolin rev 1368537001).
    assert expand_templates("{{langx|it|mandolino}}") == "mandolino"
    assert expand_templates("{{langx|de|link=no|Bratsche}}") == "Bratsche"


def test_expand_templates_floruit_matches_circas_shape():
    # {{floruit}}, real (Lute, rev 1361649316) -- always invoked bare in
    # the sample, e.g. "Francesco Spinacino ({{floruit}} 1507)".
    assert expand_templates("{{floruit}}") == "fl."
    assert expand_templates("{{floruit|1507}}") == "fl. 1507"


def test_expand_templates_siglo_bare_form_has_no_word():
    # {{siglo|XVII}}, real (Guitarra clásica, rev 174381319). Confirmed
    # against the live template via MediaWiki's own expandtemplates API,
    # not guessed: the bare form renders as just the numeral.
    assert expand_templates("{{siglo|XVII}}") == "XVII"


def test_expand_templates_siglo_lowercase_style_positional_and_named():
    # {{siglo|XVI||s}}, real (Violín, rev 174121807); {{Siglo|XVI|3=s}},
    # the equivalent named form, also real in the same article.
    assert expand_templates("{{siglo|XVI||s}}") == "siglo XVI"
    assert expand_templates("{{Siglo|XVI|3=s}}") == "siglo XVI"


def test_expand_templates_siglo_uppercase_style_capitalizes_the_word():
    # {{Siglo|XIX||S}}, real (Violonchelo, rev 173779642) -- the case of
    # the third arg is the signal MediaWiki itself reads; matched
    # case-sensitively here for exactly that reason.
    assert expand_templates("{{Siglo|XIX||S}}") == "Siglo XIX"


def test_expand_templates_formatnum_parser_function_emits_the_number():
    # {{formatnum:3000}}, real (Mandolino, rev 150920416) -- MediaWiki's
    # colon-separated parser-function syntax, not the pipe-separated
    # name|args shape every other template in this module assumes.
    assert expand_templates("a {{formatnum:3000}} km") == "a 3000 km"


def test_expand_templates_apostrophe_escape_is_not_a_literal_quote_yet():
    # expand_templates alone must not produce a literal "'" -- see the
    # module comment above _DISPLAY_TEMPLATES for why a direct substitution
    # this early in the pipeline corrupts unrelated prose. The placeholder
    # only becomes a real apostrophe after restore_apostrophe_placeholders
    # runs, post-convert_emphasis (see the round-trip tests below).
    assert "'" not in expand_templates("l{{'}}amore")


def test_wikitext_to_markdown_apostrophe_escape_does_not_corrupt_later_bold():
    # The exact real construct (violino-it.md, violoncello-it.md,
    # mandolino-it.md all use this identical l{{'}}''word'' idiom) and the
    # defect it would cause without the placeholder mechanism: a literal
    # apostrophe directly before a real ''italic'' span forms an artificial
    # three-quote run that sends convert_emphasis's BOLD regex hunting for
    # the next unrelated '''...''' anywhere later in the article, eating
    # every real sentence in between as fake bold content.
    raw = (
        "Sono famosi l{{'}}''abete di risonanza'' della Val di Fiemme. "
        "Il '''violino''' è uno strumento musicale."
    )
    body = wikitext_to_markdown(raw, title="Test", targets={})
    assert "l'*abete di risonanza*" in body
    assert "**violino**" in body
    assert "\x00" not in body


def test_restore_apostrophe_placeholders_after_expand_templates_bare_case():
    raw = "l{{'}}arte"
    assert restore_apostrophe_placeholders(expand_templates(raw)) == "l'arte"


def test_expand_templates_is_case_and_whitespace_insensitive_on_the_name():
    assert expand_templates("{{ Convert | 4 | ft }}") == "4 ft"


def test_expand_templates_leaves_an_unallowlisted_template_for_strip_templates():
    # {{cite web|...}} is chrome, not display -- expand_templates leaves it
    # completely untouched so strip_templates removes it whole, same as
    # before this fix existed.
    raw = "{{cite web|url=https://example.com|title=Example}}"
    assert expand_templates(raw) == raw
    assert strip_templates(expand_templates(raw)) == ""


def test_expand_templates_leaks_an_unclosed_convert_verbatim():
    # This module's invariant: never delete text not positively identified
    # as markup. expand_templates can't match a template with no closing
    # `}}` at all, so it falls through untouched to strip_templates's own
    # unclosed-brace handling, which leaks it verbatim.
    raw = "The body is {{convert|14|in|cm and more text with no close"
    assert expand_templates(raw) == raw
    assert strip_templates(expand_templates(raw)) == raw


def test_expand_templates_drops_named_args_only_leaves_positional_untouched():
    assert expand_templates("{{convert|4|ft|abbr=on|sp=us}}") == "4 ft"


def test_wikitext_to_markdown_expands_a_real_convert_measurement():
    # {{convert|13|in|cm}} is a real Cello measurement (rev 1360203898).
    raw = "The body of a cello is about {{convert|13|in|cm}} long."
    body = wikitext_to_markdown(raw, title="Cello", targets={})
    assert "13 in" in body
    assert "{{" not in body


def test_wikitext_to_markdown_expands_a_real_music_symbol_in_context():
    # {{music|flat}} used inline in real prose, the kind of fact an
    # exact_term gold-span query targets.
    raw = "The instrument is tuned to B{{music|flat}} major."
    body = wikitext_to_markdown(raw, title="Test", targets={})
    assert "B♭ major" in body


# --- numeric-fraction-named templates: spec-review defect 2 -----------------
#
# Wikipedia also has standalone templates literally *named* `3/4`, `1/2`,
# `1/4` -- the same inline-fraction concept as {{frac}}, under a different
# name. Real (Double bass, rev 1365602377): "the more common {{3/4}} size
# bass ... such as a {{1/2}} size or {{1/4}} size ... a {{1/2}} bass is not
# half the length". Missed on the first pass because `_DISPLAY_TEMPLATES`
# is a fixed-name dict and none of these names were in the six-article
# convert/frac/lang/circa/music/nowrap survey -- caught by a second review
# reading the committed trial output, not by re-deriving the allowlist.


def test_expand_templates_numeric_fraction_named_template():
    assert expand_templates("{{3/4}}") == "3/4"
    assert expand_templates("{{1/2}}") == "1/2"
    assert expand_templates("{{1/4}}") == "1/4"


def test_expand_templates_numeric_fraction_named_template_inline_with_prose():
    # "{{1/4}}-inch cable", real (Double bass) -- the template sits directly
    # against surrounding punctuation with no separating space.
    assert expand_templates("a {{1/4}}-inch cable") == "a 1/4-inch cable"


def test_wikitext_to_markdown_expands_the_real_double_bass_size_sentence():
    # The exact double-bass.md:32 construct the spec review flagged as
    # nonsense before this fix: "the more common  size bass" with both
    # numeric-fraction templates and the trailing {{frac|4|4}} silently
    # emptied.
    raw = (
        'Whereas the traditional "full-size" ({{frac|4|4}} size) bass stands '
        "on average {{convert|74.8|in|cm}}, the more common {{3/4}} size bass "
        "stands on average {{convert|71.6|in|cm}}. Other sizes are also "
        "available, such as a {{1/2}} size or {{1/4}} size; a {{1/2}} bass is "
        "not half the length of a {{frac|4|4}} bass."
    )
    body = wikitext_to_markdown(raw, title="Double bass", targets={})
    assert "the more common 3/4 size bass" in body
    assert "such as a 1/2 size or 1/4 size" in body
    assert "a 1/2 bass is not half the length of a 4/4 bass" in body
    assert "{{" not in body


def test_wikitext_to_markdown_expands_the_real_viola_and_hyphen_joiner_sentence():
    # The exact viola.md:22 construct the spec review flagged: "and(-)"
    # wasn't a _CONVERT_JOINERS key, so the second value and unit were
    # silently dropped and the literal joiner text leaked into the prose.
    raw = (
        "A full-size viola's body is between {{convert|25|and|100|mm|in|0|abbr=on}} "
        "longer than the body of a full-size violin (i.e., between "
        "{{convert|38|and(-)|46|cm|in|abbr=on|disp=sqbr}}), with an average "
        "length of {{convert|41|cm|in|abbr=on}}."
    )
    body = wikitext_to_markdown(raw, title="Viola", targets={})
    assert "between 38 and 46 cm" in body
    assert "and(-)" not in body


# --- dropped-template reporting: spec-review defect 3 ------------------------
#
# The systemic fix behind both defects above: a display template not on the
# allowlist is indistinguishable from real citation chrome by shape alone,
# so it vanishes with no signal. `dropped` makes that visible instead of
# relying on someone reading 79 converted files by eye.


def test_expand_templates_tallies_an_unallowlisted_name_into_dropped():
    dropped: Counter[str] = Counter()
    expand_templates("{{cite web|url=x|title=y}}", dropped)
    assert dropped == {"cite web": 1}


def test_expand_templates_tallies_are_case_and_whitespace_normalized():
    dropped: Counter[str] = Counter()
    expand_templates("{{Cite Web|x=1}} {{ cite web |y=2}}", dropped)
    assert dropped == {"cite web": 2}


def test_expand_templates_does_not_tally_an_allowlisted_or_fraction_name():
    dropped: Counter[str] = Counter()
    expand_templates("{{convert|4|ft}} {{frac|1|2}} {{3/4}}", dropped)
    assert dropped == {}


def test_expand_templates_does_not_double_count_across_convergence_passes():
    # Regression for the overcounting bug this fix's first draft had: a
    # chrome template sitting alongside a *nested* display template
    # (convert wrapping frac) needs several convergence passes before the
    # text is stable. Naively tallying inside that loop would count the
    # untouched chrome template once per pass instead of once per
    # occurrence.
    dropped: Counter[str] = Counter()
    text = "{{cite web|x=1}} {{convert|4|{{frac|1|2}} ft}} {{convert|5|ft}}"
    result = expand_templates(text, dropped)
    assert dropped == {"cite web": 1}
    assert result == "{{cite web|x=1}} 4 1/2 ft 5 ft"


def test_expand_templates_tallies_a_nested_unallowlisted_template_once_for_the_outer_name():
    dropped: Counter[str] = Counter()
    expand_templates("{{cite book|quote={{something unhandled}}}}", dropped)
    assert dropped == {"cite book": 1}


def test_expand_templates_does_not_tally_an_unclosed_template_left_leaked():
    # An unclosed template is leaked verbatim by strip_templates, not
    # removed -- it should not appear in a report of what was dropped.
    dropped: Counter[str] = Counter()
    expand_templates("{{cite web|url=x unclosed", dropped)
    assert dropped == {}


def test_expand_templates_tallies_an_unrecognized_music_argument_by_its_own_key():
    # Unlike an unallowlisted template *name*, an unrecognized *argument* to
    # an allowlisted one (music) is only visible to that handler -- it
    # empties the span immediately rather than leaving `{{...}}` behind for
    # the post-convergence sweep to find, so it has to tally itself.
    dropped: Counter[str] = Counter()
    expand_templates("{{music|breve}}", dropped)
    assert dropped == {"UNKNOWN-MUSIC-ARG:breve": 1}


def test_wikitext_to_markdown_passes_dropped_through_to_the_caller():
    dropped: Counter[str] = Counter()
    raw = "{{cite web|url=x}} A **violin** is a chordophone."
    wikitext_to_markdown(raw, title="Test", targets={}, dropped=dropped)
    assert dropped == {"cite web": 1}


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


# --- convert_links / strip_external_links / wikitext_to_markdown: Task 3 ----


def test_convert_links_rewrites_in_corpus_links_with_the_original_label():
    raw = "The [[Violin]] and the [[Double bass|contrabass]]."
    assert (
        convert_links(raw, TARGETS) == "The [[violin|Violin]] and the [[double-bass|contrabass]]."
    )


def test_convert_links_drops_out_of_corpus_links_to_plain_text():
    assert convert_links("A [[Spruce]] top.", TARGETS) == "A Spruce top."


def test_convert_links_keeps_the_link_trail_in_the_label():
    assert convert_links("two [[violin]]s", TARGETS) == "two [[violin|violins]]"


def test_convert_links_ignores_the_section_part_of_a_target():
    assert convert_links("see [[Violin#Tuning|tuning]]", TARGETS) == "see [[violin|tuning]]"


def test_convert_links_emits_the_bare_form_when_label_equals_slug():
    assert convert_links("a [[violin]] b", TARGETS) == "a [[violin]] b"


def test_strip_external_links_keeps_the_label_only():
    assert (
        strip_external_links("see [https://x.org the site] and [https://y.org]")
        == "see the site and "
    )


def test_wikitext_to_markdown_end_to_end():
    raw = (
        "{{Infobox instrument|name=Violin}}\n"
        "The '''violin''' is a [[String instrument|string instrument]].<ref>Smith</ref>\n"
        "\n"
        "== Construction ==\n"
        "[[File:Violin.jpg|thumb|A [[violin]]]]\n"
        "It has a [[Spruce]] top and is tuned in perfect fifths.\n"
        "\n"
        "* four strings\n"
        "** tuned G, D, A, E\n"
        "\n"
        "== References ==\n"
        "<references/>\n"
    )

    assert wikitext_to_markdown(raw, title="Violin", targets=TARGETS) == (
        "# Violin\n"
        "\n"
        "The **violin** is a [[string-instrument|string instrument]].\n"
        "\n"
        "## Construction\n"
        "\n"
        "It has a Spruce top and is tuned in perfect fifths.\n"
        "\n"
        "- four strings\n"
        "  - tuned G, D, A, E\n"
    )


def test_wikitext_to_markdown_produces_just_the_heading_when_the_body_strips_to_nothing():
    # An article that's all chrome (template + citation, no prose) must still
    # produce a valid note, not a trailing blank line or a crash — exercises
    # the final `.rstrip()` path with an empty stripped body.
    raw = "{{Infobox instrument|name=X}}\n<ref>Smith</ref>\n"
    assert wikitext_to_markdown(raw, title="Ghost", targets=TARGETS) == "# Ghost\n"


def test_wikitext_to_markdown_drops_a_trailing_category_link_inside_a_kept_section():
    # Item 1's fix, exercised through the full pipeline rather than
    # `convert_links` in isolation: a category link at the end of a section
    # that's otherwise kept must vanish without taking any of the real prose
    # around it with it.
    raw = (
        "The '''violin''' is a string instrument.\n"
        "\n"
        "== Construction ==\n"
        "It has four strings.[[ Category:String instruments ]]\n"
    )
    result = wikitext_to_markdown(raw, title="Violin", targets=TARGETS)
    assert "Category" not in result
    assert "It has four strings." in result


def test_wikitext_to_markdown_rewrites_a_wikilink_inside_a_heading():
    # The stated reason `convert_links` runs after `convert_headings`: only
    # unit-level coverage existed for this before (feeding pre-converted
    # `## `-heading text straight to `convert_links`), never proof that the
    # full pipeline's heading conversion happens first and the link inside
    # survives to be rewritten.
    raw = "The instrument.\n\n== The [[Violin]] Family ==\nMore text.\n"
    result = wikitext_to_markdown(raw, title="Strings", targets=TARGETS)
    assert "## The [[violin|Violin]] Family" in result


def test_wikitext_to_markdown_does_not_leak_a_resolved_link_from_a_dropped_section():
    # A "See also" section is dropped whole by `drop_sections`, which runs
    # *after* `convert_links`. A resolved, in-corpus wikilink sitting inside
    # it must still be dropped along with the rest of the section, not
    # survive because it was already rewritten to a "real" wikilink by then.
    raw = "Body text.\n\n== See also ==\n* [[Double bass]]\n"
    result = wikitext_to_markdown(raw, title="Violin", targets=TARGETS)
    assert "double-bass" not in result
    assert "Double bass" not in result


# --- real-input hazards probed for Task 3 ------------------------------------


def test_convert_links_no_label_section_link_drops_the_hash_fragment_from_the_alias():
    # The plan's snippet falls back to the raw `target` (page + "#Section")
    # for the display text when no label is given. That's a real defect: for
    # an *in-corpus* section link it leaks "#Tuning" into the alias text
    # (`[[violin|Violin#Tuning]]`), which is not natural prose. Use the
    # section-stripped page for the fallback instead.
    assert convert_links("see [[Violin#Tuning]]", TARGETS) == "see [[violin|Violin]]"


def test_convert_links_same_page_section_link_with_no_label_keeps_the_hash_as_text():
    # The mirror-image hazard: a same-page section link like [[#Construction]]
    # has an *empty* page (everything is the "#Section" part). Naively
    # stripping the hash for the display fallback would turn this into an
    # empty string, silently deleting real reader-visible link text — the
    # exact thing the module's invariant forbids. When the page component is
    # empty, fall back to the untouched target instead. It's still an
    # out-of-corpus link (an empty page can't resolve against `targets`), so
    # it degrades to plain text like any other out-of-corpus link — just with
    # its "#Construction" text preserved instead of deleted.
    assert convert_links("see [[#Construction]] below", TARGETS) == "see #Construction below"


def test_convert_links_pipe_trick_empty_label_falls_back_to_the_page():
    # `[[Violin|]]` is MediaWiki's "pipe trick" syntax. Real MediaWiki doesn't
    # just repeat the page name for it — it also strips parenthetical
    # disambiguators, namespaces, and comma suffixes (`[[Violin
    # (instrument)|]]` renders "Violin", not "Violin (instrument)"). This
    # function only does the simple case (repeat the page name); that's fine
    # because MediaWiki's pre-save transform expands the pipe trick into a
    # literal label the moment an editor saves the page, so fetched article
    # wikitext essentially never contains an empty pipe like this one.
    assert convert_links("a [[Violin|]] b", TARGETS) == "a [[violin|Violin]] b"


def test_convert_links_unresolved_unclosed_bracket_does_not_eat_a_later_paragraph():
    # Same hazard class flagged twice already in this module (see
    # test_strip_html_tags_catch_all_does_not_cross_a_newline and
    # test_strip_refs_unclosed_ref_does_not_eat_prose_up_to_the_next_ref): an
    # unclosed `[[` must not let the lazy target/label capture cross a blank
    # line hunting for some unrelated later `]]`, swallowing whole paragraphs
    # in between. The genuine link further down must still convert normally.
    raw = "See [[Violin and more text on this line.\n\nA whole separate paragraph.\n\n[[Spruce]] survives."
    assert convert_links(raw, TARGETS) == (
        "See [[Violin and more text on this line.\n\nA whole separate paragraph.\n\nSpruce survives."
    )


def test_convert_links_category_link_renders_as_nothing_not_its_label():
    # A category link is invisible in rendered article text — it files the
    # page under the category, it doesn't render as a link at all. Flattening
    # it to plain text like an ordinary out-of-corpus link would inject
    # "Category: ..." straight into the prose. Localized aliases (it/es)
    # covered too, since this corpus draws from those editions.
    assert convert_links("Strings.[[Category:String instruments]]", TARGETS) == "Strings."
    assert convert_links("Corde.[[Categoria:Strumenti a corda]]", TARGETS) == "Corde."
    assert convert_links("Cuerdas.[[Categoría:Instrumentos]]", TARGETS) == "Cuerdas."


def test_convert_links_leading_colon_forces_a_category_link_to_render_normally():
    # `[[:Category:Strings]]` (leading colon) is the real wikitext escape
    # that turns an otherwise-invisible category link into an ordinary,
    # visible inline link — the colon must not be swept up by the
    # invisible-category check above. It's still an out-of-corpus link (the
    # Category namespace isn't a corpus article), so — like any other
    # out-of-corpus link — it degrades to its plain rendered label rather
    # than becoming a wikilink; the fix here is only that the label keeps
    # the full "Category:Strings" text instead of vanishing like the
    # non-colon case does.
    assert convert_links("See [[:Category:Strings]] for a list.", TARGETS) == (
        "See Category:Strings for a list."
    )


def test_convert_links_leading_colon_file_link_is_an_ordinary_inline_link():
    # [[:File:X.jpg|the scan]] (leading colon) is a real inline link to the
    # file's description page, not an embed — `strip_media_links` deliberately
    # leaves it alone (its prefix pattern requires File: to follow directly
    # after `[[`, not after `[[:`), so it reaches convert_links and flattens
    # like any other out-of-corpus link, keeping its label.
    assert convert_links("a [[:File:X.jpg|the scan]] b", TARGETS) == "a the scan b"


def test_convert_links_media_link_is_an_ordinary_inline_link():
    assert convert_links("[[Media:song.ogg|listen]] here", TARGETS) == "listen here"


# --- quality-review round 2: whitespace, template-collapsed labels, schemes,
# reorder pin, underscores ----------------------------------------------------


def test_convert_links_category_check_tolerates_surrounding_whitespace_in_the_target():
    # MediaWiki trims wikilink target whitespace before parsing the
    # namespace, so `[[ Category:X ]]` is valid, real wikitext (and does
    # occur) — `_CATEGORY` is `^`-anchored against the *raw* regex capture,
    # which still has its surrounding whitespace, so the leading space
    # defeats the anchor and the category link leaks straight into prose
    # instead of vanishing. Fixed by stripping the target before the check.
    assert convert_links("Strings.[[ Category:String instruments ]]", TARGETS) == "Strings."
    # A space on *either* side of the colon is also valid, real wikitext.
    assert convert_links("Strings.[[ Category : String instruments ]]", TARGETS) == "Strings."


def test_convert_links_leading_colon_check_tolerates_surrounding_whitespace_too():
    # Same anchor-vs-untrimmed-capture bug, for the leading-colon override:
    # without stripping first, `[[ :Category:X ]]` leaks a literal leading
    # colon into the display (`:Category:X`) instead of being recognized as
    # the colon-forced-visible case and stripped to `Category:X`.
    assert convert_links("See [[ :Category:Strings ]] for a list.", TARGETS) == (
        "See Category:Strings for a list."
    )


def test_convert_links_whitespace_only_label_falls_back_instead_of_vanishing():
    # A template inside a link's label — a common idiom (`{{lang|it|...}}`,
    # `{{nowrap|...}}`, `{{sic}}`) — is reduced to bare whitespace by
    # `strip_templates`, six stages before `convert_links` ever runs. A label
    # of "  " is truthy, so `label or fallback` never falls back, and then
    # `.strip()` empties it — silently deleting reader-visible text for both
    # the in-corpus and out-of-corpus cases. `strip_templates` isn't run here
    # directly; the whitespace-only label is constructed by hand to isolate
    # the defect in `convert_links` itself, since that's the stage actually
    # responsible for the fix.
    assert convert_links("A [[Violin| ]] here.", TARGETS) == "A [[violin|Violin]] here."
    assert convert_links("A [[Spruce| ]] top.", TARGETS) == "A Spruce top."


def test_convert_links_never_returns_an_empty_display_for_a_resolved_link():
    # Direct invariant check (not just the specific whitespace-label repro
    # above): whenever a link resolves against `targets` — the visible,
    # in-corpus case — the rendered text must never come out empty, no
    # matter what arrived in the label after five earlier stripping stages.
    for raw in ("[[Violin|]]", "[[Violin| ]]", "[[Violin|   ]]", "[[Violin]]"):
        result = convert_links(raw, TARGETS)
        assert result.strip() != ""


def test_convert_links_underscored_target_displays_with_spaces():
    # MediaWiki renders `_` as a space in an unpiped link's visible text —
    # `_` is just the URL-safe stand-in for a space in a page title.
    # Underscored targets show up whenever wikitext is pasted from a URL
    # (`https://en.wikipedia.org/wiki/Violin_(music)` -> `[[Violin_(music)]]`).
    # The lookup already normalized underscores away; the display text didn't.
    assert convert_links("a [[Violin_(music)]] b", TARGETS) == "a Violin (music) b"


def test_media_links_before_wikilinks_keeps_the_caption_out_of_prose():
    # This is the order `wikitext_to_markdown` actually uses.
    raw = "[[File:Violin.jpg|thumb|300px|A violin resting on a stand.]] Real prose follows."
    text = strip_media_links(raw)
    text = convert_links(text, TARGETS)
    assert text == " Real prose follows."


def test_wikilinks_before_media_links_leaks_the_caption_into_prose():
    # Demonstrates why the order matters: `convert_links`'s `_LINK` pattern
    # has no notion of File-link multi-parameter syntax (`|thumb|300px|...`).
    # Its label group has no reason to stop at a second or third pipe, so it
    # greedily swallows the whole "thumb|300px|caption" run as if it were an
    # ordinary link's label. "File:Violin.jpg" resolves to nothing in
    # `targets`, so that raw "thumb|300px" fragment gets flattened straight
    # into the prose instead of being removed by `strip_media_links` — which
    # never gets the chance, because the nested brackets it needs to depth-
    # scan are already gone.
    raw = "[[File:Violin.jpg|thumb|300px|A violin resting on a stand.]] Real prose follows."
    text = convert_links(raw, TARGETS)
    assert "thumb|300px" in text


def test_strip_external_links_unterminated_link_does_not_cross_a_paragraph_break():
    # Same newline-crossing hazard as above, for the *other* link stripper:
    # `[^\]]*` has no line bound, so an unterminated `[http://...` could scan
    # past a blank line hunting for some unrelated later `]` — e.g. a stray
    # "]" that's just ordinary prose punctuation in the next paragraph — and
    # eat everything in between as the "label". Left unmatched instead.
    raw = "See [https://x.org broken and\n\nA new paragraph with a stray ] bracket kept intact."
    assert strip_external_links(raw) == raw


def test_strip_external_links_supports_protocol_relative_urls():
    # `//x.org` with no scheme at all is real wikitext (MediaWiki accepts a
    # protocol-relative URL as external-link syntax); already supported by
    # the original pattern, but never exercised by a test.
    assert strip_external_links("[//x.org the mirror]") == "the mirror"


def test_strip_external_links_supports_non_http_schemes():
    # The original pattern only recognized `http(s)://`, contradicting this
    # module's own stated design principle (see `_INLINE_TAG`'s comment: a
    # catch-all beats a hand-maintained allowlist). `ftp://`, `mailto:` and
    # friends are real MediaWiki external-link protocols; without the fix,
    # a non-http link survives untouched, leaking raw `[ftp://... label]`
    # bracket syntax into the corpus.
    assert strip_external_links("[ftp://ftp.example.org the archive]") == "the archive"
    assert strip_external_links("[mailto:info@example.org contact us]") == "contact us"


def test_strip_external_links_leaves_bracketed_editorial_notes_alone():
    # The highest-frequency non-match in real prose: an editorial `[sic]` in
    # a quotation, or a leftover citation-style `[1]`. Neither starts with a
    # scheme, so the widened pattern must still leave both alone.
    assert strip_external_links("The violin [sic] was old.") == "The violin [sic] was old."
    assert strip_external_links("A famous claim.[1]") == "A famous claim.[1]"


def test_strip_external_links_double_bracketed_url_leaves_stray_brackets():
    # Accepted, not fixed (per review): `[[https://x.org]]` isn't a real
    # wikilink (MediaWiki wikilinks don't take URLs), but the way the two
    # patterns tile means `_EXT_LINK` can match starting at the *second* `[`
    # (its own `[url]` shape fits from there), consuming through the *first*
    # `]`, and leaving the outer `[` and the second `]` behind as orphaned
    # literal characters. Real but rare; the fix would cost more than the
    # symptom, so this is documented, not corrected.
    assert strip_external_links("[[https://x.org]]") == "[]"


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


# --- shapes the allowlist had NOT already seen ------------------------------
#
# Every expansion test above pins a construct someone had already observed in
# a fetched article, which is exactly why three defects reached the corpus:
# an unrecognized `convert` joiner dropped half a measurement, a one-argument
# `{{frac}}` produced a plausible wrong number, and an allowlisted template
# wrapping an unallowlisted one lost its prose. These pin the *unrecognized*
# cases instead. Expected renderings verified against MediaWiki's
# `action=expandtemplates`.


def test_convert_renders_the_dash_by_and_plus_minus_range_joiners():
    assert expand_templates("{{convert|70|-|74|g|oz|abbr=on}}") == "70–74 g"
    assert expand_templates("{{convert|1|by|2|ft}}") == "1 by 2 ft"
    assert expand_templates("{{convert|5|+/-|1|mm}}") == "5 ± 1 mm"


def test_convert_keeps_both_values_and_reports_an_unrecognized_joiner():
    """The failure this guards against is silent and wrong rather than
    merely missing: the old code emitted the joiner token in place of the
    unit and discarded the second value, so `70 -` read as a measurement."""
    dropped: Counter[str] = Counter()
    assert expand_templates("{{convert|9|zzz|11|kg}}", dropped) == "9 zzz 11 kg"
    assert dropped == {"UNKNOWN-CONVERT-JOINER:zzz": 1}


def test_convert_does_not_mistake_an_output_unit_for_a_range():
    """`{{convert|4|ft|m}}` has three positional args like a range does, but
    the third is a unit, not a number -- which is how the two are told apart."""
    assert expand_templates("{{convert|4|ft|m}}") == "4 ft"


def test_frac_with_one_argument_is_a_denominator_not_a_whole_number():
    assert expand_templates("{{frac|2}}") == "1/2"


def test_frac_adjacent_to_a_digit_keeps_a_separator():
    """`19{{frac|2}}` is nineteen and a half. Without the separator the flat
    text reads "191/2" -- the banjo scale-length defect."""
    assert expand_templates("19{{frac|2}} to 21{{frac|2}} inches") == "19 1/2 to 21 1/2 inches"


def test_a_positional_argument_containing_an_equals_sign_is_kept_as_text():
    assert expand_templates("{{blockquote|Foo said x = y here.|Author}}") == (
        '"Foo said x = y here." — Author'
    )


def test_the_numeric_escape_places_its_value_positionally():
    assert expand_templates("{{nowrap|1=a = b}}") == "a = b"


def test_a_blocked_expansion_is_reported_distinctly_from_ordinary_chrome():
    """An allowlisted template whose argument holds an unallowlisted one is
    never innermost, so it survives to be stripped whole -- taking its prose
    with it. In a report dominated by hundreds of `cite book` lines, a bare
    `langx` entry reads as chrome; the BLOCKED prefix is what makes it
    findable."""
    dropped: Counter[str] = Counter()
    expand_templates("{{blockquote|A quotation.{{sfn|Smith|2001}}}}", dropped)
    assert dropped == {"BLOCKED:blockquote": 1}


def test_a_handler_that_swallows_its_argument_reports_it():
    dropped: Counter[str] = Counter()
    expand_templates("{{blockquote|}}{{nowrap|}}", dropped)
    assert dropped == {}  # no arguments to lose
    expand_templates("{{music|zzz}}", dropped)
    assert dropped == {"UNKNOWN-MUSIC-ARG:zzz": 1}


def test_blockquote_keeps_a_positional_author_beside_a_named_quote():
    assert expand_templates("{{blockquote|text=Q|Author}}") == '"Q" — Author'


def test_decode_entities_resolves_nbsp_but_cannot_invent_markup():
    assert decode_entities("Op.&nbsp;9") == "Op. 9"
    # Decoding runs after tag stripping precisely so this stays inert text.
    assert decode_entities("&lt;div&gt;") == "<div>"


def test_tidy_punctuation_clears_brackets_left_holding_nothing():
    assert tidy_punctuation("The cello ( , ), also called") == "The cello, also called"
    assert tidy_punctuation("The cello (a bowed instrument)") == "The cello (a bowed instrument)"


def test_a_nul_in_the_source_cannot_impersonate_the_apostrophe_placeholder():
    assert "\x00" not in wikitext_to_markdown("a\x00b ''c''", title="T", targets={})


def test_a_script_wrapper_keeps_the_text_it_wraps():
    """`{{script/Hebr|...}}` styles non-Latin prose; nesting it inside
    `{{langx}}` blocked that expansion and deleted the Hebrew term from
    harp.md — the first defect the BLOCKED report line surfaced."""
    assert expand_templates("{{langx|he|{{script/Hebr|כִּנּוֹר}}}}") == "כִּנּוֹר"
