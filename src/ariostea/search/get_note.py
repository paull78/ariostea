from __future__ import annotations

from ariostea.domain.models import NoteMissing, NoteText
from ariostea.ports.store import DocumentReader
from ariostea.ports.vault import NoteUnreadable, Vault


class GetNote:
    """A note's file as written, for a path the index holds. The exact index
    lookup comes first, so a path the index never stored (a traversal, a
    private file) is never read (vault-hardening.09)."""

    def __init__(self, reader: DocumentReader, vault: Vault) -> None:
        self._reader = reader
        self._vault = vault

    def get(self, path: str) -> NoteText | NoteMissing:
        title = self._reader.note_titles([path]).get(path)
        if title is None:
            return NoteMissing(path=path, reason=None)
        try:
            file = self._vault.read(path)
        except NoteUnreadable as exc:
            return NoteMissing(path=path, reason=exc.reason)
        return NoteText(path=path, title=title, text=file.text)
