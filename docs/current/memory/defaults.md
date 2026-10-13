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
