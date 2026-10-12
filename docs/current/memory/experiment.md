# Experiment design

### embedding-sweep.01 · decision · 2026-10-12 · spec
The sweep measures three candidates against `paraphrase-multilingual-mpnet-base-v2`: `intfloat/multilingual-e5-large`, `BAAI/bge-m3` and `jinaai/jina-embeddings-v3`. jina-v3 can become the default only on a separate licence call, because it is CC BY-NC 4.0.
**Why:** measuring jina costs about one extra hybrid hour and nothing else, and leaving it out would leave a strong candidate unmeasured. Rejected: e5-large and bge-m3 only (saves an hour, loses the comparison); e5-large only (no custom registration, but drops bge-m3, the strongest-rated MIT option).

### embedding-sweep.02 · decision · 2026-10-12 · spec
The decision rule stays as in chunking and contextual retrieval: hybrid overall span recall at k=5 up at least 0.03 on the control, no query type down more than 0.05, at least 160 of 167 spans reachable. Ties go to span MRR, then lower cost.
**Why:** results stay comparable with every earlier run, and five extra cross-lingual cases alone move overall by 0.03, so a cross-lingual-only win can still pass. Rejected: an extra cross-lingual bar (breaks comparability, can reject an overall win); a looser rule for this sweep (threshold chosen with the experiment in view).

### embedding-sweep.03 · decision · 2026-10-12 · spec
The sweep spends what it needs: about three CPU hours for HYBRID plus indexing, and about 7 GB of model downloads in `~/.cache/fastembed`. It stops and reports after the first-stage pass only if no candidate beats the control's dense overall.
**Why:** the reranker cannot recover what dense never retrieves on cross-lingual, so a candidate with no dense gain has no route to a hybrid gain. Rejected: hybrid for every candidate whatever step 3 shows (up to three wasted hours); a pause after step 3 every time (waits on the owner for no new information when a candidate is ahead).

### embedding-sweep.06 · discovery · 2026-10-12 · step 03
The step 03 smoke run (e5-large, 160t+40, DENSE, `--no-log`, unofficial) scored dense span recall 0.772 overall against the control's 0.545, and 0.696 on cross_lingual against 0.500. Every type rose: buried 0.875, exact_term 0.732, paraphrase 0.800. 167 of 167 spans reachable, so the XLM-R tokenizer cut the corpus as the control did. Discriminated-out cases: 0.931, the same as the control.
**Why:** an early signal only. The logged first-stage run in step 07 is the measurement; the stop rule after step 07 will not trigger on this evidence.

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

### embedding-sweep.14 · discovery · 2026-10-12 · step 07
First stage, all with unit vectors, against the normalised control (DENSE 0.593, cross_lingual 0.587, FUSED 0.671): e5-large DENSE 0.778 (cross_lingual 0.652), FUSED 0.731; bge-m3 DENSE 0.856 (cross_lingual 0.891), FUSED 0.784; jina-v3 DENSE 0.832 (cross_lingual 0.870), FUSED 0.760. SPARSE is 0.533 in every run, confirming identical chunks. Discriminated-out cases: DENSE 0.966 for all three candidates against 0.931 for the control. Every candidate beats the control's dense, so the stop rule does not trigger and step 08 runs HYBRID for all three. Two side findings: for the candidates FUSED is below DENSE (sparse noise on cross_lingual drags fusion down), and jina-v3 embeds a query in 110 ms median against 15 to 17.5 ms for e5 and bge-m3. Indexing: 310 to 362 s against 111 s for the control.
**Why:** protocol step 3 and the stop rule (embedding-sweep.03, embedding-sweep.08).
