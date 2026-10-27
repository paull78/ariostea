# Step 05 · Progress callback and printer

**Milestone:** 2 · **Needs:** 04 · **Covers:** AC-16

## Goal
`IndexVault.index(progress=None)` reports every note as a plain value; the CLI prints throttled progress lines on stderr for `reindex` and for the first pass of `watch`. MCP tools print nothing.

## Files touched
- `src/ariostea/domain/models.py`: `Progress(done: int, total: int, embedded: int, skipped: int)`.
- `src/ariostea/indexing/index_vault.py`: optional `progress: Callable[[Progress], None]`; called once per listed note (unchanged, embedded or skipped) and never when absent.
- `src/ariostea/indexing/watch_vault.py`: passes an optional progress callback to the first pass only.
- `src/ariostea/progress.py` (new, driving side, next to `cli.py`): `ProgressPrinter(write, clock=time.monotonic, every=100, interval=10.0)`; prints `412/3,410 notes · 37 embedded · 2 skipped` when `done % every == 0`, or `clock() - last >= interval`, or `done == total`.
- `src/ariostea/cli.py`: `reindex` and `watch` create a printer writing to stderr.
- Tests: `tests/indexing/test_index_vault.py`, `tests/test_progress.py` (new), `tests/test_cli.py`, `tests/mcp/test_handlers.py`.

## Tests to write first
- `test_progress_reports_every_listed_note`: 5 notes (2 unchanged, 2 changed, 1 unreadable) → 5 calls, last `Progress(5, 5, 2, 1)`.
- `test_printer_prints_every_100_and_at_end`: 250 calls with a frozen clock → lines at 100, 200, 250.
- `test_printer_prints_on_interval`: clock advancing 10 s per call → a line every call (AC-16).
- `test_printer_format`: thousands separator and the middle dots as in the spec.
- `test_reindex_cli_progress_on_stderr_only` (AC-16): CliRunner, 250-note in-memory vault through a fake container; stdout holds only the summary.
- `test_mcp_reindex_passes_no_progress`: `reindex_payload` calls `index()` without a callback.

## Implementation notes
- vault-hardening.16: one plain mode, the clock lives in the driving adapter.
- `total` is `len(vault.notes())`, known before reading.
- AC-16's stdio clause: the MCP server never writes progress; `reindex_payload` passes none. The awkward-vault test in step 10 checks stdout carries only protocol messages.

## Verify
`uv run pytest -m "not integration" -q` → all pass.

## Done when
- manual: `uv run ariostea reindex --config <scratch config>` on a fresh store over the synthetic vault prints progress lines, then the summary.
