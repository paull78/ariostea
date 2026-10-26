# Step 03 · One-line CLI errors, openai_compat removed

**Milestone:** 1 · **Needs:** 02 · **Covers:** AC-14, AC-15, AC-01 (CLI exit status)

## Goal
Every CLI command turns a known error into one line on stderr and exit status 1. The `[embedding]` section no longer accepts `provider`, `base_url`, `api_key` or `model`.

## Files touched
- `src/ariostea/config/schema.py`: remove the four fields from `EmbeddingCfg`; a `model_validator(mode="before")` raises `ValueError` naming any of them if present: `embedding.<key> was removed: Ariostea embeds locally with fastembed; set embedding.local_model`. Add a `ConfigError` (in `schema.py`) that `load_config` raises for a missing file (`config file not found: <path>`), TOML syntax errors and pydantic validation errors, with a one-line message (first error's dotted location and message).
- `src/ariostea/cli.py`: one helper that runs a command body and maps `ConfigError` and `VaultUnavailable` to `typer.echo(msg, err=True)` + `raise typer.Exit(1)`. Used by `serve`, `reindex`, `watch`, `status`.
- `tests/config/test_schema.py`: the openai_compat parsing test is replaced by the removal tests.
- `tests/test_cli.py`: new cases.

## Tests to write first
- `test_removed_embedding_keys_are_rejected` (parametrised over the four keys): `load_config` raises `ConfigError` whose message names `embedding.<key>` and `local_model`.
- `test_missing_config_file_is_a_config_error`, `test_missing_vault_path_is_a_config_error`, `test_bad_toml_is_a_config_error`: one-line messages.
- `test_cli_config_error_prints_one_line_and_exits_1` (each command; CliRunner with `mix_stderr=False` or the installed Typer's equivalent): stderr is one line, no `Traceback`, exit code 1, and `build_container` was never called (AC-14: no model loaded).
- `test_cli_vault_unavailable_prints_one_line_and_exits_1` (AC-15, AC-01): fake container whose indexer raises `VaultUnavailable`.

## Implementation notes
- Pydantic's default `extra="ignore"` stays for other keys (non-goal); only the four removed keys are named.
- Do not catch broad exceptions: an unknown error should still show its traceback.
- Check the owner's untracked `ariostea.toml` and `ariostea.example.toml` load: the example still lists the removed keys until step 09, so the example-config test (if one parses it) must be updated here, or the example's four lines removed here. Remove them here and leave the README wording to step 09.

## Verify
`uv run pytest tests/config tests/test_cli.py -q` → all pass; `uv run ariostea status --config /nonexistent.toml; echo $?` → one line `config file not found: /nonexistent.toml`, then `1`.

## Done when
- No CLI path prints a traceback for a missing or invalid config, a removed key or a vault guard.
