import time, random
from fastembed.rerank.cross_encoder import TextCrossEncoder
from ariostea.adapters.fastembed_cache import fastembed_cache_dir
m = TextCrossEncoder("jinaai/jina-reranker-v2-base-multilingual", cache_dir=fastembed_cache_dir())
tok = m.model.tokenizer
text = open("README.md").read()
ids = tok.encode(text).ids
random.seed(1)
def passage(n):
    i = random.randrange(len(ids) - n)
    return tok.decode(ids[i:i+n])
q = "how do I rebuild the index after changing the model"
list(m.rerank(q, [passage(64) for _ in range(8)]))
for n in (256, 128):
    for pool in (100, 50, 20):
        docs = [passage(n) for _ in range(pool)]
        t = time.perf_counter(); list(m.rerank(q, docs)); dt = time.perf_counter() - t
        print(f"{n} tokens, pool {pool:3d}: {dt*1000:6.0f} ms", flush=True)
