# Store

The sqlite store: schema, migration of existing indexes, vector table size.

### bge-m3-default.01 · discovery · 2026-10-13 · spec
`SqliteStore` creates `chunks_vec` with `CREATE VIRTUAL TABLE IF NOT EXISTS ... FLOAT[dim]`, so an index built at one vector size keeps that size forever. Reopened at another size, the first insert fails: "Dimension mismatch for inserted vector for the "embedding" column. Expected 3 dimensions but received 4." (checked on a throwaway database). Switching from the 768-dimension default to bge-m3 at 1024 would therefore break the next reindex of every existing vault. The 2026-10-09 switch from `bge-small-en-v1.5` (384) to the multilingual model (768) had the same exposure.
**Why:** the store must rebuild the vector table when the size changes, so the fix is in scope for this cycle.

### bge-m3-default.07 · discovery · 2026-10-13 · step 05
The store drops a vector table of the wrong size when it opens, not when it indexes. `ariostea serve` and `ariostea status` also open the store, and neither reindexes, so after the default changes a server started before any reindex answers from sparse retrieval alone, and `status` reports an empty fingerprint, until `ariostea reindex` (or the `reindex` MCP tool) runs. Before this cycle the same situation made every dense query fail with a dimension mismatch.
**Why:** the upgrade note in the README and example config (step 06) must tell users to run `ariostea reindex` once after upgrading. Rejected for now: reindexing automatically on `serve` when the fingerprint is empty (a behaviour change outside this cycle's scope).

### bge-m3-default.09 · issue · 2026-10-13 · gate M2
The gate's hands-on reindex of the owner's vault failed: "OperationalError: table chunks has no column named context_blurb". The owner's index (59 notes, 377 chunks, fingerprint without a chunker part) predates the September blurb work: its `chunks` table has no `context_blurb` column, and it has no `chunks_fts` or `meta` table. `_init_schema` only runs `CREATE TABLE IF NOT EXISTS`, which creates the missing tables (empty) but never adds a column to an existing one, so any index older than the blurb cycle fails its next reindex on any model. Step 01's rebuild ran before the failure, so the vectors were dropped and the fingerprint cleared. The owner's config and index were restored from scratchpad backups; status shows the old fingerprint again.
**Why:** blocks the spec's behaviour for existing vaults ("re-embeds every note, no error, no manual step"); needs a fix step before gate M2 can pass.
