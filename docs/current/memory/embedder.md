# Embedder adapter

### embedding-sweep.04 · discovery · 2026-10-12 · step 01
Two step 01 tests passed before any implementation: `test_control_fingerprint_is_unchanged` and `test_count_tokens_ignores_the_prefix`. Both guard behaviour the change had to keep, not add. A deliberate break (a suffix on the fingerprint, the passage prefix inside `count_tokens`) made both fail, so they are real guards.
**Why:** recorded per the build rule on tests born green; no action needed.
