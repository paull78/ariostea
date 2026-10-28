"""Shared fixtures."""

import pytest


@pytest.fixture
def old_index():
    """Build an index with the schema of one created before September 2026."""

    def _build(path, dim):
        """A database with the schema of an index built before September 2026,
        copied from a real one: no `context_blurb` column, no FTS table, no meta
        table (bge-m3-default.09)."""
        import sqlite3

        import sqlite_vec

        db = sqlite3.connect(path)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.executescript(
            f"""
            CREATE TABLE notes (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                mtime REAL NOT NULL
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                heading_path TEXT NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL
            );
            CREATE VIRTUAL TABLE chunks_vec USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{dim}]
            );
            """
        )
        db.commit()
        db.close()

    return _build


@pytest.fixture
def previous_release_index():
    """Build an index as the release before server-robustness left it."""

    def _build(path, dim, notes, fingerprint):
        """`notes` maps each path to its content hash; `fingerprint` is the
        index-wide one that release stored in `meta`. The schema is copied from
        `SqliteStore._init_schema` at commit 4c7d0dc."""
        import sqlite3

        import sqlite_vec

        db = sqlite3.connect(path)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.executescript(
            f"""
            CREATE TABLE notes (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                mtime REAL NOT NULL
            );
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,
                note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                heading_path TEXT NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                context_blurb TEXT
            );
            CREATE VIRTUAL TABLE chunks_vec USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{dim}]
            );
            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text,
                tokenize="unicode61 remove_diacritics 2"
            );
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        for note_path, content_hash in notes.items():
            db.execute(
                "INSERT INTO notes(path, title, content_hash, mtime) VALUES (?, ?, ?, 0)",
                (note_path, note_path, content_hash),
            )
        db.execute("INSERT INTO meta(key, value) VALUES ('fingerprint', ?)", (fingerprint,))
        db.commit()
        db.close()

    return _build
