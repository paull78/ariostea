from ariostea.eval.wikitext import (
    strip_comments,
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


def test_strip_templates_handles_nesting():
    assert strip_templates("a {{convert|4|{{frac|1|2}} ft}} b") == "a  b"


def test_strip_tables_removes_the_whole_table():
    assert strip_tables("x\n{| class=wikitable\n|-\n| cell\n|}\ny") == "x\n\ny"


def test_strip_media_links_removes_captions_including_nested_links():
    assert strip_media_links("[[File:V.jpg|thumb|A [[violin]] on a table]]Text") == "Text"
    assert strip_media_links("[[Immagine:V.jpg|thumb|foto]]Testo") == "Testo"


def test_strip_media_links_leaves_ordinary_links_alone():
    assert strip_media_links("a [[violin]] b") == "a [[violin]] b"


def test_strip_html_tags_removes_galleries_and_inline_tags():
    assert strip_html_tags("<small>a</small><gallery>\nFile:x.jpg\n</gallery>b") == "ab"
