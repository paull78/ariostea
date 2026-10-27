# Step 09 · Status over MCP and CLI

**Milestone:** 2 · **Needs:** 06, 08 · **Covers:** AC-02, AC-10, AC-11

## Goal
`status`, over MCP and from the CLI, reports `notes`, `chunks`, `last_indexed` (ISO 8601 UTC or null), `config_fingerprint` (last completed run, or null), `running`, `last_run` (`finished_at`, `notes`, `chunks`, `skipped`, `error`, or null), and `progress` only for a run inside this server. The integration tests prove it across processes.

## Files touched
- `src/ariostea/mcp/handlers.py`: `status_payload(container, background=None)` builds the fields from `container.admin.stats()` and `container.runs.state()`.
- `src/ariostea/mcp/server.py`: `status` and `reindex` docstrings list the fields.
- `src/ariostea/cli.py`: `status` prints the same payload (as today, one dict).
- `tests/mcp/test_handlers.py`, `tests/test_cli.py`: unit tests below.
- `tests/test_server_robustness.py`: the cross-process tests below.

## Tests to write first
Unit, with fakes:
- `test_status_on_a_new_index`: `last_indexed`, `config_fingerprint`, `last_run` are None, `running` False, no `progress`.
- `test_status_after_a_completed_run`: `last_indexed == "2026-10-27T09:41:05Z"` for a fixed `now`, `config_fingerprint` the run's, `last_run` with its counts and `error: None`.
- `test_status_after_a_failed_run` (AC-02): `last_run.error` set; `last_indexed` and `config_fingerprint` from the earlier completed run.
- `test_status_is_stable_between_runs` (AC-10 third part): two calls with no run between give the same `last_indexed`.

Integration (`tests/test_server_robustness.py`):
- `test_status_during_and_after_a_server_run` (AC-02): extends the AC-01 test: during the run `running: true` and `progress.done <= progress.total`; after, `running: false`, `last_run` with notes, chunks, skipped (the awkward vault skips its bad files) and `finished_at`, and no `progress`.
- `test_server_sees_a_cli_run` (AC-10, AC-11): a server is started on a fresh store and `status` gives `last_indexed: null`; a CLI `reindex` runs in a subprocess; while it runs, the server's `status` shows `running: true` and no `progress`; after it exits, `last_indexed` is within 1 s of the moment the subprocess ended (the CLI is timed from the test), `config_fingerprint` is non-empty and `last_run` matches the CLI's printed counts.
- `test_guard_failure_is_reported` (AC-02): with the vault folder renamed away on an indexed store, MCP `reindex` returns started; `status` then shows `last_run.error` mentioning the vault, `last_indexed` unchanged, and the note count unchanged. The folder is renamed back afterwards.

## Implementation notes
- `finished_at` and `last_indexed` format: `record.finished_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")`; one helper in `handlers.py`.
- `progress` key present only when `background` gives a value.
- The CLI `status` prints the payload as today; no new formatting.

## Verify
`uv run pytest tests/mcp tests/test_cli.py -q` → all pass; `uv run pytest tests/test_server_robustness.py -q` → all pass.

## Done when
- The new tests pass; the status payload has exactly the fields in the spec.
- manual: at the final gate, read `status` from a client during and after a `reindex`, and after a CLI run.
