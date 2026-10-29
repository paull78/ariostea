import time, random
import numpy as np
from fastembed.rerank.cross_encoder import TextCrossEncoder
from ariostea.adapters.fastembed_cache import fastembed_cache_dir
name = "jinaai/jina-reranker-v2-base-multilingual"
base = TextCrossEncoder(name, cache_dir=fastembed_cache_dir())
tok = base.model.tokenizer
ids = tok.encode(open("README.md").read() + open("ARCHITECTURE.md").read()).ids
random.seed(2)
docs = [tok.decode(ids[i:i+256]) for i in (random.randrange(len(ids)-256) for _ in range(50))]
q = "how do I rebuild the index after changing the model"
ref = np.array(list(base.rerank(q, docs)))
for opts in ({}, {"MLComputeUnits": "ALL", "ModelFormat": "MLProgram"}, {"MLComputeUnits": "CPUAndGPU", "ModelFormat": "MLProgram"}):
    try:
        m = TextCrossEncoder(name, cache_dir=fastembed_cache_dir(), providers=[("CoreMLExecutionProvider", opts), "CPUExecutionProvider"])
        list(m.rerank(q, docs[:4]))
        for _ in range(2):
            t = time.perf_counter(); s = np.array(list(m.rerank(q, docs))); dt = time.perf_counter() - t
        print(f"coreml {opts}: {dt*1000:.0f} ms, max score diff {np.abs(s-ref).max():.4f}", flush=True)
    except Exception as e:
        print(f"coreml {opts}: failed {type(e).__name__}: {str(e)[:200]}", flush=True)
