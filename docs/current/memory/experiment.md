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
