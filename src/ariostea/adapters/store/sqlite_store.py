from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path
from typing import Sequence

import sqlite_vec

from ariostea.domain.models import (
    Chunk,
    ContextualizedChunk,
    IndexStats,
    Note,
    NoteDocument,
    QueryFilters,
    RetrievedChunk,
)
from ariostea.ports.store import ChunkRetriever, DocumentReader, DocumentWriter, IndexAdmin


def _fts_query(text: str) -> str:
    r"""Turn free user text into a safe FTS5 MATCH expression.

    FTS5 MATCH is its own query language; raw punctuation or an empty string
    raises a syntax error. We extract word tokens and OR them so any term may
    match (recall-first — fusion handles precision).

    The token pattern is Unicode-aware (``\w`` matches accented letters), so
    words like "città" or "Müller" reach FTS intact. FTS5 then applies the
    table tokenizer (unicode61 + remove_diacritics) to fold accents on both the
    query and the index, making keyword search accent-insensitive.
    """
    terms = re.findall(r"\w+", text, re.UNICODE)
    return " OR ".join(f'"{t}"' for t in terms)


class SqliteStore(DocumentWriter, DocumentReader, ChunkRetriever, IndexAdmin):
    def __init__(self, path: str, dim: int) -> None:
        self._dim = dim
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                mtime REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                note_id INTEGER NOT NULL REFERENCES notes(id) ON DELETE CASCADE,
                ordinal INTEGER NOT NULL,
                heading_path TEXT NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                context_blurb TEXT
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[{self._dim}]
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                chunk_id UNINDEXED,
                text,
                tokenize="unicode61 remove_diacritics 2"
            );
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            """
        )
        self.db.commit()

    # --- DocumentWriter ---
    def upsert_note(
        self,
        note: Note,
        chunks: Sequence[ContextualizedChunk],
        embeddings: Sequence[list[float]],
    ) -> None:
        cur = self.db.cursor()
        cur.execute("BEGIN")
        try:
            self._delete_note_rows(cur, note.path)
            cur.execute(
                "INSERT INTO notes(path, title, content_hash, mtime) VALUES (?,?,?,?)",
                (note.path, note.title, note.content_hash, note.mtime),
            )
            note_id = cur.lastrowid
            for cc, vec in zip(chunks, embeddings):
                ch = cc.chunk
                cur.execute(
                    "INSERT INTO chunks(note_id, ordinal, heading_path, text, token_count, context_blurb) "
                    "VALUES (?,?,?,?,?,?)",
                    (note_id, ch.ordinal, "/".join(ch.heading_path), ch.text, ch.token_count, cc.context_blurb),
                )
                chunk_id = cur.lastrowid
                cur.execute(
                    "INSERT INTO chunks_vec(chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, sqlite_vec.serialize_float32(vec)),
                )
                cur.execute(
                    "INSERT INTO chunks_fts(chunk_id, text) VALUES(?, ?)",
                    (chunk_id, cc.embedding_text),
                )
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def delete_note(self, path: str) -> None:
        cur = self.db.cursor()
        cur.execute("BEGIN")
        try:
            self._delete_note_rows(cur, path)
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def _delete_note_rows(self, cur: sqlite3.Cursor, path: str) -> None:
        row = cur.execute("SELECT id FROM notes WHERE path = ?", (path,)).fetchone()
        if row is None:
            return
        note_id = row["id"]
        chunk_ids = [
            r["id"]
            for r in cur.execute("SELECT id FROM chunks WHERE note_id = ?", (note_id,)).fetchall()
        ]
        for cid in chunk_ids:
            cur.execute("DELETE FROM chunks_vec WHERE chunk_id = ?", (cid,))
            cur.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
        cur.execute("DELETE FROM chunks WHERE note_id = ?", (note_id,))
        cur.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    # --- ChunkRetriever ---
    def dense(
        self, vec: list[float], k: int, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]:
        rows = self.db.execute(
            """
            WITH knn AS (
                SELECT chunk_id, distance
                FROM chunks_vec
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT c.note_id, n.path AS note_path, c.ordinal, c.heading_path,
                   c.text, c.token_count, knn.distance
            FROM knn
            JOIN chunks c ON c.id = knn.chunk_id
            JOIN notes n ON n.id = c.note_id
            ORDER BY knn.distance
            """,
            (sqlite_vec.serialize_float32(vec), k),
        ).fetchall()
        results: list[RetrievedChunk] = []
        for rank, r in enumerate(rows):
            heading_path = tuple(p for p in r["heading_path"].split("/") if p)
            chunk = Chunk(
                note_path=r["note_path"],
                ordinal=r["ordinal"],
                heading_path=heading_path,
                text=r["text"],
                token_count=r["token_count"],
            )
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=1.0 / (1.0 + r["distance"]),
                    dense_rank=rank,
                    sparse_rank=None,
                )
            )
        return results

    def sparse(
        self, query: str, k: int, filters: QueryFilters | None = None
    ) -> list[RetrievedChunk]:
        match = _fts_query(query)
        if not match:
            return []
        rows = self.db.execute(
            """
        WITH bm AS (
            SELECT chunk_id, bm25(chunks_fts) AS bm
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY bm
            LIMIT ?
        )
        SELECT n.path as note_path, c.ordinal, c.heading_path, c.text, c.token_count, bm.bm
        FROM bm
        JOIN chunks c ON c.id = bm.chunk_id
        JOIN notes n ON n.id = c.note_id
        ORDER BY bm.bm
        """,
            (match, k),
        ).fetchall()
        results: list[RetrievedChunk] = []
        for rank, r in enumerate(rows):
            heading_path = tuple(p for p in r["heading_path"].split("/") if p)
            chunk = Chunk(
                note_path=r["note_path"],
                ordinal=r["ordinal"],
                heading_path=heading_path,
                text=r["text"],
                token_count=r["token_count"],
            )
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    score=-r["bm"],
                    dense_rank=None,
                    sparse_rank=rank,
                )
            )
        return results

    # --- DocumentReader ---
    def note_titles(self, paths: Sequence[str]) -> dict[str, str]:
        paths = list(paths)
        if not paths:
            return {}
        placeholders = ",".join("?" for _ in paths)
        rows = self.db.execute(
            f"SELECT path, title FROM notes WHERE path IN ({placeholders})", paths
        ).fetchall()
        return {r["path"]: r["title"] for r in rows}

    def read_note(self, path) -> NoteDocument | None:
        note = self.db.execute("SELECT title FROM notes WHERE path = ?", (path,)).fetchone()
        if note is None:
            return None
        rows = self.db.execute(
            "SELECT c.text FROM chunks c JOIN notes n ON n.id = c.note_id "
            "WHERE n.path = ? ORDER BY c.ordinal",
            (path,),
        ).fetchall()
        body = "\n\n".join(r["text"] for r in rows)
        return NoteDocument(note_path=path, title=note["title"], text=body)

    # --- IndexAdmin ---
    def known_hashes(self) -> dict[str, str]:
        rows = self.db.execute("SELECT path, content_hash FROM notes").fetchall()
        return {r["path"]: r["content_hash"] for r in rows}

    def stats(self) -> IndexStats:
        notes = self.db.execute("SELECT COUNT(*) AS c FROM notes").fetchone()["c"]
        chunks = self.db.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"]
        return IndexStats(
            notes=notes,
            chunks=chunks,
            last_indexed=time.time(),
            config_fingerprint=self.fingerprint(),
        )

    def fingerprint(self) -> str:
        row = self.db.execute("SELECT value FROM meta WHERE key = 'fingerprint'").fetchone()
        return row["value"] if row else ""

    def set_fingerprint(self, value: str) -> None:
        self.db.execute(
            "INSERT INTO meta(key, value) VALUES ('fingerprint', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (value,),
        )
        self.db.commit()
