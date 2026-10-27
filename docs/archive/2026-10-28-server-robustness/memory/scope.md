# Scope

What the cycle fixes and what it leaves out.

### server-robustness.01 · decision · 2026-10-27 · spec
The cycle fixes the four server problems vault-hardening deferred: the MCP `reindex` tool blocking the server, an interrupted index re-embedding every note, "database is locked" between concurrent writers, and `status.last_indexed` reporting the current time.
**Why:** all four are about who writes the index and when, so they share code; together they are what still breaks daily use on a large vault. Causes found in the code before the interview: FastMCP calls sync tools on its event loop; the fingerprint is written only at the end of a run, so a restarted run sees a mismatch and re-embeds everything; sqlite runs in its default rollback journal with Python's 5 s busy timeout; `SqliteStore.stats()` returns `time.time()`. Rejected: blocking and resume only (leaves the lock and the wrong time, which touch the same store code); adding other deferred items (unused config keys, symlinks, `.canvas`/`.base`), unrelated to writing.

### server-robustness.06 · decision · 2026-10-27 · spec
Non-goals: cancelling a run, queueing MCP `reindex` calls, progress for runs in other processes, a `status` count of notes on other settings, Windows, the rest of vault-hardening's deferred list, retrieval changes.
**Why:** the owner kept the cycle to the four fixes. The three extras offered (old-settings count, cross-process progress, cancel) were each cheap to describe but add state or tools no fix needs.

### server-robustness.11 · decision · 2026-10-27 · plan
The owner approved the outline and asked to build through to the end with the recommended option at every choice and a final recap. Gate 1 is taken as approved in sequential mode; each milestone gate and the final gate still run their checks and show their recap, then carry on. Stand-ins for the owner's calls are listed in the recap.
**Why:** the owner's instruction ("implement it till done use recommended choices and make a final recap"). Sequential over parallel: the two parallel pairs are small and touch neighbouring code, so worktrees would cost more than they save.
