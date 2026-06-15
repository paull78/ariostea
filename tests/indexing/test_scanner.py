from ariostea.indexing.scanner import scan_vault


def test_scan_finds_markdown_and_hashes(tmp_path):
    (tmp_path / "a.md").write_text("# A\nhello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("# B\nworld")
    (tmp_path / "ignore.txt").write_text("not markdown")

    found = {f.rel_path: f for f in scan_vault(tmp_path, ignore=[])}
    assert set(found) == {"a.md", "sub/b.md"}
    assert found["a.md"].raw.startswith("# A")
    assert len(found["a.md"].content_hash) == 64  # sha256 hex


def test_scan_respects_ignore(tmp_path):
    (tmp_path / "keep.md").write_text("k")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "config.md").write_text("x")
    found = {f.rel_path for f in scan_vault(tmp_path, ignore=[".obsidian/"])}
    assert found == {"keep.md"}
