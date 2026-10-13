# Retrieval and evaluation

Decisions and discoveries about retrieval quality, the eval protocol and the embedding model. Results tables live in `docs/retrieval-tuning.md`.

### embedding-sweep.02 · decision · 2026-10-12 · spec
The decision rule stays as in chunking and contextual retrieval: hybrid overall span recall at k=5 up at least 0.03 on the control, no query type down more than 0.05, at least 160 of 167 spans reachable. Ties go to span MRR, then lower cost.
**Why:** results stay comparable with every earlier run, and five extra cross-lingual cases alone move overall by 0.03, so a cross-lingual-only win can still pass. Rejected: an extra cross-lingual bar (breaks comparability, can reject an overall win); a looser rule for this sweep (threshold chosen with the experiment in view).

### embedding-sweep.07 · issue · 2026-10-12 · gate M1
The dense channel ranks by sqlite-vec's default metric, Euclidean distance, and fastembed returns unnormalised vectors for the control (norms about 2.2 to 2.4) and for e5 (about 30). So every logged dense number ranked by raw Euclidean distance, not cosine, the similarity these models were trained for. bge-m3 as planned (normalisation on) and jina-v3 return unit vectors, so the sweep as specified compares unevenly. A scratch rerun with unit vectors, unlogged, gave dense span recall at 5: control 0.593 overall and 0.587 cross_lingual (raw: 0.545 and 0.500); e5 0.778 and 0.652 (raw: 0.772 and 0.696). Effect on HYBRID unknown.
**Why:** a confound in the approved protocol and a likely production defect; changing the control's vectors breaks AC-01 and forces a reindex, so it goes to the owner at gate M1.

### embedding-sweep.08 · pivot · 2026-10-12 · gate M1
The cycle adds an `embedding.normalize` setting, off by default, which makes the local embedder return unit vectors. The sweep measures the control raw and normalised, and every candidate normalised. Candidates are judged against the normalised control; the normalised control is judged against the raw control by the same rule. One extra HYBRID hour. Turning normalisation on by default is the owner's call after the sweep, together with the model. Responds to embedding-sweep.07.
**Why:** keeps production and AC-01 unchanged while removing the uneven comparison, and measures normalisation and model as separate factors. Rejected: always normalise now (breaks AC-01 and forces a reindex before any measurement); keep the spec as is (bge-m3 and jina-v3 would be compared on unit vectors against a raw control).

### embedding-sweep.12 · discovery · 2026-10-12 · step 06
Pilot (`eval/results/logs/2026-10-12-embedding-pilot.txt`): all four models load with unit vectors and none failed. Every tokenizer cuts the corpus at 160t+40 into the same 4,246 chunks, so the sweep varies the model alone. e5 (prefixes) and jina-v3 (fastembed task ids) embed a query and a passage of the same text differently (cosine 0.946 and 0.934); the control and bge-m3 embed them identically, as expected. Load / 100 chunks: control 0.5 s / 2.5 s, e5 2.2 s / 6.9 s, bge-m3 21.7 s / 6.7 s, jina-v3 14.3 s / 7.7 s, so the large models embed about 2.7 to 3 times slower than the control.
**Why:** all three candidates survive to step 07; no issue entries needed.

### embedding-sweep.13 · discovery · 2026-10-12 · step 06
The raw control reproduces exactly at the current commit (run `2026-10-12-chunk-160t+40-dense-fused-sparse`): DENSE 0.545, SPARSE 0.533, FUSED 0.629, as logged in September. The logged HYBRID control `2026-09-24-chunk-160t+40-dense-hybrid-sparse` (0.880) therefore stands, and no HYBRID rerun was needed. The normalised control (`2026-10-12-chunk-160t+40-norm-dense-fused-sparse`): DENSE 0.593 (cross_lingual 0.587), SPARSE 0.533 (unchanged, as expected), FUSED 0.671 (cross_lingual 0.435 against 0.304 raw). Indexing took about 111 s either way; query embedding median 4.8 to 4.9 ms.
**Why:** settles protocol step 1; the normalised control is the comparison point for steps 07 and 08 (embedding-sweep.08).

### embedding-sweep.15 · discovery · 2026-10-12 · step 08
HYBRID span recall at k=5, hits out of 167 (cross_lingual out of 46): raw control 147 (32), 0.880; normalised control 150 (35), 0.898; e5-large 154 (39), 0.922; bge-m3 155 (41), 0.928; jina-v3 155 (41), 0.928. Every candidate loses one buried case (40 to 39, −0.025, within the −0.05 limit); exact_term is 39 of 41 everywhere; paraphrase is 36 of 40 except e5 at 37. Span MRR: raw 0.810, normalised 0.823, e5 0.848, bge-m3 0.846, jina-v3 0.848. Note recall: bge-m3 0.988, jina-v3 0.982, e5 0.952, normalised 0.964. Discriminated-out cases: 1.000 for every candidate. Cost: indexing 303 s (bge-m3), 313 s (e5), 359 s (jina-v3) against 109 s; query embedding median 14.3 ms (bge-m3), 14.7 ms (e5), 116.3 ms (jina-v3) against 4.8 ms. A HYBRID run took 11 to 16 minutes, not the hour the plan assumed.
**Why:** protocol step 4 measurements, before the verdicts in embedding-sweep.16.

### embedding-sweep.17 · discovery · 2026-10-12 · step 08
The 256t+64 follow-up for bge-m3 (DENSE and SPARSE, unit vectors, run `2026-10-12-chunk-256t+64-bge-m3-norm-dense-sparse`) beats bge-m3 at 160t+40 on both channels: DENSE 0.880 against 0.856 (cross_lingual 0.891 both; exact_term 0.805 against 0.756; paraphrase 0.925 against 0.875), SPARSE 0.575 against 0.533. 167 of 167 spans reachable. HYBRID was not measured at 256t+64. It ran on the five-case reading of embedding-sweep.16 because it is cheap and outside the decision; it does not settle that reading.
**Why:** the 160-token default was tuned for the control's 128-token training length (design doc, Confounds); a larger model may want larger chunks, and this is the first evidence it does. A chunking re-tune for a new default would be its own cycle.

### embedding-sweep.18 · decision · 2026-10-12 · final gate
The owner reads the 0.03 threshold as five cases out of 167, the intent the chunking design gave it. Under that reading bge-m3 and jina-v3 pass the decision rule against the normalised control (+5 cases each, no type down more than 0.025, 167 of 167 reachable). e5-large (+4) and normalisation alone (+3) still fail.
**Why:** the threshold was always described as the smallest change the set can resolve, which is five cases; 5/167 is 0.0299 only because 0.03 was a rounded stand-in. The reading was made after the data, by the owner, and is recorded as such. Rejected: the strict numeric reading, under which nothing passes.

### embedding-sweep.19 · decision · 2026-10-12 · final gate
The next cycle switches the default embedding model to `BAAI/bge-m3` (MIT, unit vectors natively, 14 ms per query), measuring HYBRID at 256t+64 first since the follow-up pass favoured larger chunks. Every vault reindexes once. The fastembed cache fix (embedding-sweep.10) goes into that cycle or one beside it. jina-v3 is not adopted: same recall, eight times the query latency, non-commercial licence.
**Why:** bge-m3 passes the rule (embedding-sweep.18) and gains 8 cases (cross_lingual +9 of 46) over what users run today. Rejected: normalisation only (+3 cases, fails the rule); keeping the defaults (leaves the measured gain unused).

# Embedder adapter

### bge-m3-default.03 · decision · 2026-10-13 · spec
The eval's pinned model, `MULTILINGUAL_MODEL` in `src/ariostea/eval/wiki_index.py`, moves to `BAAI/bge-m3`. Run ids without a model slug mean bge-m3 from this cycle on; the old model runs with `--model sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, and past runs keep their recorded `embedding` field.
**Why:** a plain eval run should measure the configuration users get, so the next experiment starts from the new default. Rejected: keeping the old model pinned (past runs reproduce without flags, but every future experiment would need `--model` and a plain run would measure a model nobody ships).

### bge-m3-default.06 · decision · 2026-10-13 · step 04
The default chunk size becomes 256 model tokens with overlap 64. bge-m3's HYBRID run at 256t+64 (`2026-10-13-chunk-256t+64-bge-m3-norm-hybrid`) found 157 of 167 against 155 at 160t+40: buried 40 of 40 (+1), paraphrase 37 of 40 (+1), cross_lingual 41 of 46 and exact_term 39 of 41 unchanged. Span MRR 0.867 against 0.846, note recall 0.988 both, all 167 spans reachable, discriminated-out cases 1.000. Indexing 334 s, query embedding median 15.2 ms. Against the configuration users run today (147 hits, 0.880) the new defaults find 10 more cases, span recall 0.940.
**Why:** passes the rule in bge-m3-default.02 (at least 155, no type down more than 2 cases) with room to spare.

### bge-m3-default.08 · discovery · 2026-10-13 · step 05
With bge-m3 as the default, the full suite takes 22 min 46 s against about 12 minutes before, because the integration tests that build the default container now embed with the larger model; the fast suite is unchanged at under a second. `test_old_model_run_ids_carry_its_slug` passed before the switch: naming a model's slug in the run id was built in the embedding-sweep cycle, and the test guards it now that the old model is the one that needs naming. Hard-coded references to the old model that stay on purpose: `CONTROL` in the adapter tests (a model without prefixes), `eval/run_eval.py`, `eval/run_contextual_eval.py` and their integration tests (older eval sets with their own pinned model), and the blurb experiment's control run ids.
**Why:** records the cost of the switch on the test loop and why the remaining references are not stale.
