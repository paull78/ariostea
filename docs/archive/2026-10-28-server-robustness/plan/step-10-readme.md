# Step 10 · README

**Milestone:** 2 · **Needs:** 08 · **Covers:** AC-13

## Goal
The README describes the new behaviour: `reindex` starts a run and returns, `status` lists its fields, one run at a time (refused over MCP, waited for by the CLI and `watch`), interrupted runs resume, and deleting the index means `rm index.db*` with nothing running, with the `.lock` file mentioned.

## Files touched
- `README.md`: the tools table, the indexing and rebuild paragraphs.

## Tests to write first
Exception: prose; checked by the verify command and by reading it.

## Implementation notes
- Follow `~/.claude/VOICE.md` via `human-write-time`, neutral register (de-slop only): no first person, no em or en dashes as punctuation, digits for metrics, sentence-case headings, British spelling.
- Places to change today: the tools table rows for `status` and `reindex` (around line 83); the paragraph that says to delete the index file for a full rebuild (around line 144); the paragraph on running `reindex` before connecting a client (around line 153).
- Say that a change of model or chunking now resumes if interrupted, and that search quality is mixed until it finishes (server-robustness.04).
- Mention the `-wal`, `-shm` and `.lock` files next to the index, and that `.lock` stays after a run (it is only locked while a run goes on).

## Verify
`grep -n "rm .*index.db\*" README.md && grep -n "started" README.md && ! grep -n "delete the index file and" README.md` → the first two print a line each, the third prints nothing.

## Done when
- Every place that says to delete the index names the `-wal` and `-shm` files.
- manual: read the changed README sections.
