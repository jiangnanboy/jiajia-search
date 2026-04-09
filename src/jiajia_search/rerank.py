from fastembed.rerank.cross_encoder import TextCrossEncoder
from fastembed.common.model_description import ModelSource
from .config import config
from .utils import sigmoid, get_relevance_level

# 🔥 全局唯一模型缓存（程序生命周期只加载1次）
_rerank_model_instance = None

def load_rerank_model():
    """
    🔥 全局单例加载重排模型：只加载一次，永久复用
    解决：每次调用都重新加载模型的性能问题
    """
    global _rerank_model_instance
    # 模型已加载 → 直接返回缓存（0耗时）
    if _rerank_model_instance is not None:
        return _rerank_model_instance

    # 首次调用 → 加载模型（仅1次）
    if not config.RERANK_MODEL_PATH:
        raise ValueError("❌ 请先配置重排模型路径：config.setup(...)")

    TextCrossEncoder.add_custom_model(
        model=config.RERANK_MODEL_PATH,
        model_file=config.RERANK_ONNX_FILE,
        sources=ModelSource(url=config.RERANK_MODEL_PATH),
    )
    _rerank_model_instance = TextCrossEncoder(model_name=config.RERANK_MODEL_PATH, specific_model_path=config.RERANK_MODEL_PATH)
    print("✅ 重排模型【已预加载】→ 全局复用中...")
    return _rerank_model_instance

def rerank_pairs(query: str, docs: list[str]) -> list[float]:
    """独立调用：原始重排logits分数（模型已缓存）"""
    model = load_rerank_model()
    pairs = [(query, doc) for doc in docs]
    return list(model.rerank_pairs(pairs))

def rerank_with_normalize(query: str, docs: list[str]) -> tuple[dict, dict]:
    """独立调用：重排+归一化+相关性等级（模型已缓存）"""
    raw_scores = rerank_pairs(query, docs)
    norm_scores = [sigmoid(s) for s in raw_scores]
    rel_levels = [get_relevance_level(s) for s in norm_scores]
    return dict(zip(docs, norm_scores)), dict(zip(docs, rel_levels))