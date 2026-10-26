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


def test_note_with_only_fenced_hashes_titles_from_file_name():  # AC-11
    raw = '```python\n# This comment is not a title\nprint("hello")\n```\n\nBody text.\n'
    note, _ = ObsidianMarkdownParser().parse("Tech/Scripts/Code first.md", raw, mtime=0.0)
    assert note.title == "Code first"


def test_real_title_above_code_wins():  # AC-11
    raw = "# Deploy notes\n\n```bash\n# not the title\n```\n"
    note, _ = ObsidianMarkdownParser().parse("d.md", raw, mtime=0.0)
    assert note.title == "Deploy notes"


def test_title_after_a_fence_is_found():
    raw = "```\n# code\n```\n\n# Real title\n"
    note, _ = ObsidianMarkdownParser().parse("d.md", raw, mtime=0.0)
    assert note.title == "Real title"
