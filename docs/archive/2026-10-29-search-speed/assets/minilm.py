import time, random, os
os.environ.pop("HF_TOKEN", None)
from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastembed.common.model_description import ModelSource
from ariostea.adapters.fastembed_cache import fastembed_cache_dir
from huggingface_hub import hf_hub_download
base = TextCrossEncoder("jinaai/jina-reranker-v2-base-multilingual", cache_dir=fastembed_cache_dir())
tok = base.model.tokenizer
ids = tok.encode(open("README.md").read() + open("ARCHITECTURE.md").read()).ids
random.seed(2)
docs = [tok.decode(ids[i:i+256]) for i in (random.randrange(len(ids)-256) for _ in range(50))]
q = "how do I rebuild the index after changing the model"
repo = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
for f in ("onnx/model.onnx", "onnx/model_qint8_arm64.onnx"):
    for extra in (f, "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        hf_hub_download(repo, extra, cache_dir=fastembed_cache_dir(), token=False)
    name = "local/mminilm-" + f.split("/")[-1]
    TextCrossEncoder.add_custom_model(model=name, sources=ModelSource(hf=repo), model_file=f, size_in_gb=0.5)
    m = TextCrossEncoder(name, cache_dir=fastembed_cache_dir())
    list(m.rerank(q, docs[:4]))
    t = time.perf_counter(); list(m.rerank(q, docs)); dt = time.perf_counter() - t
    print(f"RESULT {f}: pool 50 x ~256t {dt*1000:.0f} ms", flush=True)
