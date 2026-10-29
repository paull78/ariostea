# Issues

### search-speed.18 · issue · 2026-10-28 · step 04
The Hugging Face token stored on this machine (`~/.cache/huggingface/token`) has expired, and huggingface_hub sends it even for public repositories, so any model fastembed has not cached yet fails to download with a 401 and the reranker falls back to fused order. Cached models (bge-m3, jina) are unaffected. The step 04 integration test sets `huggingface_hub.constants.HF_HUB_DISABLE_IMPLICIT_TOKEN` so it tests the fetch, not the credential; the step 05 sweep runs with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`. The product is unchanged: a user with an expired token gets the same fallback and warning as for any download failure. Open; the owner can refresh the token with `hf auth login` or remove it.
**Why:** found when the empty-cache test failed with a 401; the credential is the owner's and is left alone.

### search-speed.23 · issue · 2026-10-29 · gate M2
At pool 30, `search_sources` (k=10) returns fewer than 10 notes for 101 of 167 wiki queries, against 54 at jina pool 100. The missing notes are the low-ranked tail. A later fix: give `search_sources` its own pool of 50, which with mMiniLM costs about 1 s per call (p95 1059 ms at pool 50) while `search` stays at 707 ms.
**Why:** the owner accepted the cost at the Milestone 2 gate (search-speed.22); the separate pool was rejected in search-speed.12 only because it cost 2.3 s with jina, which no longer holds.
