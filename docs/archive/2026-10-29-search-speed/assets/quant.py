import time, random, os
os.environ.pop("HF_TOKEN", None)
import numpy as np
from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastembed.common.model_description import ModelSource
from ariostea.adapters.fastembed_cache import fastembed_cache_dir
base = TextCrossEncoder("jinaai/jina-reranker-v2-base-multilingual", cache_dir=fastembed_cache_dir())
tok = base.model.tokenizer
ids = tok.encode(open("README.md").read() + open("ARCHITECTURE.md").read()).ids
random.seed(2)
docs = [tok.decode(ids[i:i+256]) for i in (random.randrange(len(ids)-256) for _ in range(50))]
q = "how do I rebuild the index after changing the model"
ref = np.array(list(base.rerank(q, docs)))
for f in ("onnx/model_int8.onnx", "onnx/model_quantized.onnx", "onnx/model_fp16.onnx"):
    name = "local/jina-v2-" + f.split("_")[-1].split(".")[0]
    TextCrossEncoder.add_custom_model(model=name, sources=ModelSource(hf="jinaai/jina-reranker-v2-base-multilingual"), model_file=f, size_in_gb=0.3)
    m = TextCrossEncoder(name, cache_dir=fastembed_cache_dir())
    list(m.rerank(q, docs[:4]))
    t = time.perf_counter(); s = np.array(list(m.rerank(q, docs))); dt = time.perf_counter() - t
    rho = np.corrcoef(np.argsort(np.argsort(-ref)), np.argsort(np.argsort(-s)))[0, 1]
    top = len(set(np.argsort(-ref)[:5]) & set(np.argsort(-s)[:5]))
    print(f"{f}: pool 50 x 256t {dt*1000:.0f} ms, rank corr {rho:.3f}, top-5 overlap {top}/5", flush=True)
