import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import PoolingType, ModelSource
from .config import config

# 🔥 全局唯一模型缓存（程序生命周期只加载1次）
_embedding_model_instance = None

def load_embedding_model():
    """
    🔥 全局单例加载向量模型：只加载一次，永久复用
    解决：每次调用都重新加载模型的性能问题
    """
    global _embedding_model_instance
    # 模型已加载 → 直接返回缓存（0耗时）
    if _embedding_model_instance is not None:
        return _embedding_model_instance

    # 首次调用 → 加载模型（仅1次）
    if not config.VECTOR_MODEL_PATH:
        raise ValueError("❌ 请先配置向量模型路径：config.setup(...)")

    TextEmbedding.add_custom_model(
        model=config.VECTOR_MODEL_PATH,
        pooling=PoolingType.MEAN,
        normalization=True,
        sources=ModelSource(url=config.VECTOR_MODEL_PATH),
        dim=config.VECTOR_DIM,
        model_file=config.VECTOR_ONNX_FILE
    )
    _embedding_model_instance = TextEmbedding(model_name=config.VECTOR_MODEL_PATH, specific_model_path=config.VECTOR_MODEL_PATH)
    print("✅ 向量模型【已预加载】→ 全局复用中...")
    return _embedding_model_instance

def generate_embedding(text: str) -> np.ndarray:
    """独立调用：文本 → Embedding向量（模型已缓存，秒级响应）"""
    model = load_embedding_model()
    return np.array(list(model.embed(text))[0])

def generate_embeddings_batch(texts: list[str]) -> list[np.ndarray]:
    """独立调用：批量文本 → 批量Embedding向量"""
    model = load_embedding_model()
    return [np.array(emb) for emb in model.embed(texts)]