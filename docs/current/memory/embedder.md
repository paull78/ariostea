# Embedder adapter

### embedding-sweep.04 · discovery · 2026-10-12 · step 01
Two step 01 tests passed before any implementation: `test_control_fingerprint_is_unchanged` and `test_count_tokens_ignores_the_prefix`. Both guard behaviour the change had to keep, not add. A deliberate break (a suffix on the fingerprint, the passage prefix inside `count_tokens`) made both fail, so they are real guards.
**Why:** recorded per the build rule on tests born green; no action needed.

### embedding-sweep.05 · discovery · 2026-10-12 · step 03
fastembed 0.8.0 warns that `intfloat/multilingual-e5-large` "now uses mean pooling instead of CLS embedding" and suggests pinning 0.5.1 or `add_custom_model` to keep CLS. Mean pooling is what the e5 model card prescribes, so the current fastembed behaviour is the correct one and the warning needs no action.
**Why:** a run on CLS pooling would measure a misconfigured e5. Rejected: pinning fastembed 0.5.1 or registering a CLS variant, both of which restore the wrong pooling.
