# 导出核心配置（用户必用）
from .config import config

# 导出所有独立API
from .embedding import generate_embedding, generate_embeddings_batch
from .retrieval import bm25_fts_retrieval, vector_index_retrieval, parallel_hybrid_retrieval
from .fusion import rrf_fusion
from .rerank import rerank_pairs, rerank_with_normalize
from .pipeline import position_aware_blend, jiajia_search_pipeline

__version__ = "0.1.0"
__author__ = "你的名字"