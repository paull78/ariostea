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
