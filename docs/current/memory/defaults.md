# Defaults and migration

### bge-m3-default.01 · discovery · 2026-10-13 · spec
`SqliteStore` creates `chunks_vec` with `CREATE VIRTUAL TABLE IF NOT EXISTS ... FLOAT[dim]`, so an index built at one vector size keeps that size forever. Reopened at another size, the first insert fails: "Dimension mismatch for inserted vector for the "embedding" column. Expected 3 dimensions but received 4." (checked on a throwaway database). Switching from the 768-dimension default to bge-m3 at 1024 would therefore break the next reindex of every existing vault. The 2026-10-09 switch from `bge-small-en-v1.5` (384) to the multilingual model (768) had the same exposure.
**Why:** the store must rebuild the vector table when the size changes, so the fix is in scope for this cycle.

### bge-m3-default.02 · decision · 2026-10-13 · spec
256t+64 becomes the default chunk size if bge-m3's HYBRID run finds at least 155 of 167 hits (the 160t+40 result) and no query type loses more than 2 cases. Otherwise 160t+40 stays.
**Why:** the model switch already forces every vault to reindex, so a chunk-size change costs users nothing extra, and larger chunks mean fewer chunks to embed and store; the bar only has to rule out a loss. Stated in whole cases per the project lesson on thresholds. Rejected: the standard +5-case rule (built for changes that cost a reindex of their own); skipping the measurement (leaves the dense and sparse gains of embedding-sweep.17 unused).

### bge-m3-default.03 · decision · 2026-10-13 · spec
The eval's pinned model, `MULTILINGUAL_MODEL` in `src/ariostea/eval/wiki_index.py`, moves to `BAAI/bge-m3`. Run ids without a model slug mean bge-m3 from this cycle on; the old model runs with `--model sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, and past runs keep their recorded `embedding` field.
**Why:** a plain eval run should measure the configuration users get, so the next experiment starts from the new default. Rejected: keeping the old model pinned (past runs reproduce without flags, but every future experiment would need `--model` and a plain run would measure a model nobody ships).

### bge-m3-default.04 · discovery · 2026-10-13 · step 01
`test_reopening_at_the_same_dimension_keeps_vectors_and_fingerprint` passed before any implementation because it guards today's behaviour; forcing the rebuild on every open made it fail. The store reads the vector table's size from its DDL in `sqlite_master` (`FLOAT[n]`), since an empty table has no rows to measure.
**Why:** born-green note per the build rule.

### bge-m3-default.05 · discovery · 2026-10-13 · step 02
`test_reindexing_with_a_model_of_another_size_replaces_every_vector` passed on first run because step 01 already rebuilt the table. With the rebuild call removed it fails with the error a real vault would hit: "sqlite3.OperationalError: Dimension mismatch for inserted vector for the "embedding" column. Expected 2 dimensions but received 3." After the rebuild, every vector has the new size and the chunk and vector counts match the first index.
**Why:** born-green note per the build rule; the failure message confirms the test exercises the real migration path.

### bge-m3-default.06 · decision · 2026-10-13 · step 04
The default chunk size becomes 256 model tokens with overlap 64. bge-m3's HYBRID run at 256t+64 (`2026-10-13-chunk-256t+64-bge-m3-norm-hybrid`) found 157 of 167 against 155 at 160t+40: buried 40 of 40 (+1), paraphrase 37 of 40 (+1), cross_lingual 41 of 46 and exact_term 39 of 41 unchanged. Span MRR 0.867 against 0.846, note recall 0.988 both, all 167 spans reachable, discriminated-out cases 1.000. Indexing 334 s, query embedding median 15.2 ms. Against the configuration users run today (147 hits, 0.880) the new defaults find 10 more cases, span recall 0.940.
**Why:** passes the rule in bge-m3-default.02 (at least 155, no type down more than 2 cases) with room to spare.

### bge-m3-default.07 · discovery · 2026-10-13 · step 05
The store drops a vector table of the wrong size when it opens, not when it indexes. `ariostea serve` and `ariostea status` also open the store, and neither reindexes, so after the default changes a server started before any reindex answers from sparse retrieval alone, and `status` reports an empty fingerprint, until `ariostea reindex` (or the `reindex` MCP tool) runs. Before this cycle the same situation made every dense query fail with a dimension mismatch.
**Why:** the upgrade note in the README and example config (step 06) must tell users to run `ariostea reindex` once after upgrading. Rejected for now: reindexing automatically on `serve` when the fingerprint is empty (a behaviour change outside this cycle's scope).

### bge-m3-default.08 · discovery · 2026-10-13 · step 05
With bge-m3 as the default, the full suite takes 22 min 46 s against about 12 minutes before, because the integration tests that build the default container now embed with the larger model; the fast suite is unchanged at under a second. `test_old_model_run_ids_carry_its_slug` passed before the switch: naming a model's slug in the run id was built in the embedding-sweep cycle, and the test guards it now that the old model is the one that needs naming. Hard-coded references to the old model that stay on purpose: `CONTROL` in the adapter tests (a model without prefixes), `eval/run_eval.py`, `eval/run_contextual_eval.py` and their integration tests (older eval sets with their own pinned model), and the blurb experiment's control run ids.
**Why:** records the cost of the switch on the test loop and why the remaining references are not stale.

### bge-m3-default.09 · issue · 2026-10-13 · gate M2
The gate's hands-on reindex of the owner's vault failed: "OperationalError: table chunks has no column named context_blurb". The owner's index (59 notes, 377 chunks, fingerprint without a chunker part) predates the September blurb work: its `chunks` table has no `context_blurb` column, and it has no `chunks_fts` or `meta` table. `_init_schema` only runs `CREATE TABLE IF NOT EXISTS`, which creates the missing tables (empty) but never adds a column to an existing one, so any index older than the blurb cycle fails its next reindex on any model. Step 01's rebuild ran before the failure, so the vectors were dropped and the fingerprint cleared. The owner's config and index were restored from scratchpad backups; status shows the old fingerprint again.
**Why:** blocks the spec's behaviour for existing vaults ("re-embeds every note, no error, no manual step"); needs a fix step before gate M2 can pass.

### bge-m3-default.10 · discovery · 2026-10-13 · step 07
With missing `chunks` columns added on open, a scratchpad copy of the owner's old index reindexed cleanly on the new defaults: 88 notes and 952 chunks (the old index held 59 and 377, so it was months stale), fingerprint `fastembed:BAAI/bge-m3|noop|chunk:heading_aware:254:64:model_tokens`, 952 vectors all of size 1024, 952 FTS rows, in 1 min 53 s. The chunker records 254 because two of the 256 tokens go to the model's start and end markers. The shared old-schema builder became a `conftest.py` fixture, since `tests/` is not a package and one test file cannot import from another.
**Why:** retires issue bge-m3-default.09 before the real-vault check is repeated.
