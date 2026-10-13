# Recap: embedding-sweep

Feature cycle on `feat/embedding-sweep`, started and finished on 2026-10-12. Design: `docs/design/2026-10-12-embedding-sweep.md`. Results: `docs/retrieval-tuning.md`, "Embedding model".

## What was built

The cycle set out to find whether a newer multilingual embedding model fixes cross-lingual retrieval, the one query type the pipeline handled badly (hybrid span recall 0.696 against 0.90 or better elsewhere). Everything in the spec was built, plus one setting the first runs showed was missing.

- The local embedder sends documents through fastembed's passage mode and queries through its query mode, and adds the `query: ` and `passage: ` prefixes e5 needs. The control model keeps its vectors and fingerprint, so no vault reindexes.
- `BAAI/bge-m3`, absent from fastembed's list, is registered from its Hugging Face ONNX export on first use (embedding-sweep.09).
- The eval index and `eval/run_chunk_sweep.py` take a model, and every logged run records indexing time and query latency.
- `embedding.normalize`, off by default, scales vectors to length 1 so the store's Euclidean ranking equals cosine ranking (embedding-sweep.08). The runner has a matching `--normalize` flag.
- `eval/pilot_embeddings.py` checks that each candidate loads and behaves before a sweep.
- Ten logged runs, the write-up, and version 9 of the log artifact.

Small deviations: the bge-m3 load was proven in step 04 instead of step 06, and the container test goes through a `_build_embeddings` helper like its reranker sibling (embedding-sweep.11).

## How it went

Three milestones, all gates passed on 2026-10-12. Milestone 1 ran e5 through the runner. Milestone 2 added bge-m3, cost fields and the normalise setting, and reran the controls. Milestone 3 ran the sweep and wrote it up.

The pivot came at the first gate. The plan assumed the dense channel ranked by similarity, until the gate's acceptance check printed dot products far above 1: fastembed returns unnormalised vectors for the control and for e5, and sqlite-vec ranks by Euclidean distance, so every dense number ever logged had ranked raw vectors (embedding-sweep.07). bge-m3 and jina-v3 return unit vectors, so the planned comparison would have mixed two changes. The owner chose a setting off by default and a sweep that measures normalisation as its own factor, with candidates judged against the normalised control (embedding-sweep.08).

The raw control reproduced exactly at the current commit, so the September hybrid figure stood (embedding-sweep.13). The plan budgeted an hour per hybrid run; each took 11 to 16 minutes.

Hybrid hits out of 167, cross-lingual out of 46: raw control 147 and 32, normalised control 150 and 35, e5-large 154 and 39, bge-m3 155 and 41, jina-v3 155 and 41 (embedding-sweep.15). bge-m3 and jina-v3 beat the normalised control by five cases, which is 0.0299 against a rule that says 0.03. The rule was set as "about five cases", and the owner read it that way at the final gate (embedding-sweep.18).

## Decisions that stand

- embedding-sweep.01 · Three candidates measured: e5-large, bge-m3, jina-v3 · jina-v3 measured despite its non-commercial licence, adoptable only on a separate call.
- embedding-sweep.02 · Decision rule unchanged from earlier cycles · keeps results comparable.
- embedding-sweep.03 · Spend the compute, stop after the first stage only if no candidate beats the control's dense · the reranker cannot recover what dense never retrieves.
- embedding-sweep.08 · `embedding.normalize`, off by default; candidates judged against the normalised control · separates the model from the metric without a forced reindex.
- embedding-sweep.18 · The 0.03 threshold reads as five cases out of 167 · the intent it was set with; read after the data, by the owner.
- embedding-sweep.19 · The next cycle makes `BAAI/bge-m3` the default, measuring hybrid at 256t+64 first · passes the rule, +8 cases over what users run today, MIT, 14 ms per query.

## Discoveries

- fastembed's mean pooling for e5 is correct, despite its warning (embedding-sweep.05).
- All four tokenizers cut the corpus into the same 4,246 chunks, so the sweep varied the model alone (embedding-sweep.12).
- Unit vectors lift the control: dense 0.545 to 0.593, hybrid 147 to 150 cases. Normalisation alone fails the rule (embedding-sweep.16).
- With the new models, fused retrieval scores below dense alone, because sparse finds one cross-lingual case in 46 (embedding-sweep.14).
- jina-v3 embeds a query in 116 ms against 14 ms for bge-m3 (embedding-sweep.15).
- bge-m3 at 256 tokens with overlap 64 beats 160 tokens on dense (0.880 against 0.856) and sparse (0.575 against 0.533). Hybrid was not measured at that size (embedding-sweep.17).

## Open issues and known limitations

- The production server lets fastembed cache models in the system temp directory, which macOS purges and leaves broken (embedding-sweep.10). The eval pins `~/.cache/fastembed`; the server does not. Scheduled with the next cycle.
- bge-m3 sits in both caches, about 4.5 GB, because one check ran outside the eval.
- The gold set was filtered with the control model's channels, which flatters any change. Every candidate scored 1.000 on the 29 discriminated-out cases, so no easy case was lost.
- Each candidate loses one buried case. Whether it is the same case was not checked.

## Takeaways

The normalisation problem surfaced only because an acceptance check printed raw dot products, and they were far above 1. Without that number the sweep would have compared unit-vector candidates with a raw control, and the gain would have mixed the model with the metric.

Fixing the decision rule before the runs meant the five-case gap had to be settled openly, by the owner, with the data in view. Stating the threshold as five cases out of 167 would have avoided the question.

The hour-per-hybrid budget came from an older run and was about four times too high. Timing one run before planning the rest would have given a better estimate.
