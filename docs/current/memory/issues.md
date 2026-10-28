# Issues

### search-speed.18 · issue · 2026-10-28 · step 04
The Hugging Face token stored on this machine (`~/.cache/huggingface/token`) has expired, and huggingface_hub sends it even for public repositories, so any model fastembed has not cached yet fails to download with a 401 and the reranker falls back to fused order. Cached models (bge-m3, jina) are unaffected. The step 04 integration test sets `huggingface_hub.constants.HF_HUB_DISABLE_IMPLICIT_TOKEN` so it tests the fetch, not the credential; the step 05 sweep runs with `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`. The product is unchanged: a user with an expired token gets the same fallback and warning as for any download failure. Open; the owner can refresh the token with `hf auth login` or remove it.
**Why:** found when the empty-cache test failed with a 401; the credential is the owner's and is left alone.
