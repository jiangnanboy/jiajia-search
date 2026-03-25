import jieba
import concurrent.futures
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from .embedding import generate_embedding, generate_embeddings_batch

def bm25_fts_retrieval(query: str, documents: list[str]) -> list[tuple[str, int]]:
    """独立调用：纯 BM25 全文检索"""
    tokenized_docs = [list(jieba.cut(doc.strip())) for doc in documents]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = list(jieba.cut(query.strip()))
    scores = [abs(score) for score in bm25.get_scores(tokenized_query)]
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [(doc, idx+1) for idx, (doc, score) in enumerate(ranked)]

def vector_index_retrieval(
    query: str,
    documents: list[str],
    doc_embeddings: list[np.ndarray] = None
) -> list[tuple[str, int]]:
    """独立调用：纯向量语义检索（支持预生成 Embedding）"""
    query_emb = generate_embedding(query).reshape(1, -1)
    if doc_embeddings is None:
        doc_embeddings = generate_embeddings_batch(documents)
    doc_emb_matrix = np.array(doc_embeddings)
    sim_scores = cosine_similarity(query_emb, doc_emb_matrix)[0]
    ranked = sorted(zip(documents, sim_scores), key=lambda x: x[1], reverse=True)
    return [(doc, idx+1) for idx, (doc, score) in enumerate(ranked)]

def parallel_hybrid_retrieval(query: str, documents: list[str]) -> tuple[list, list]:
    """独立调用：并行 BM25 + 向量检索（加速拉满）"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_bm25 = executor.submit(bm25_fts_retrieval, query, documents)
        future_vector = executor.submit(vector_index_retrieval, query, documents)
        return future_bm25.result(), future_vector.result()