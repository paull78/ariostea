from ariostea.adapters.parse.obsidian import ObsidianMarkdownParser


def test_parses_frontmatter_title_tags_links():
    raw = (
        "---\n"
        "status: draft\n"
        "---\n"
        "# Retrieval Augmented Generation\n\n"
        "We use [[Embeddings]] and #search techniques.\n"
    )
    parser = ObsidianMarkdownParser()
    note, body = parser.parse("ideas/rag.md", raw, mtime=1.0)

    assert note.title == "Retrieval Augmented Generation"
    assert note.frontmatter == {"status": "draft"}
    assert "search" in note.tags
    assert "Embeddings" in note.wikilinks
    assert body.startswith("# Retrieval Augmented Generation")
    assert "status: draft" not in body  # frontmatter stripped


def test_title_falls_back_to_filename_when_no_heading():
    parser = ObsidianMarkdownParser()
    note, _ = parser.parse("notes/loose-thought.md", "just text, no heading", mtime=2.0)
    assert note.title == "loose-thought"
