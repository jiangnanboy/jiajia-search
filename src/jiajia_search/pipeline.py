from .retrieval import parallel_hybrid_retrieval
from .fusion import rrf_fusion
from .rerank import rerank_with_normalize
from .config import config

def position_aware_blend(rrf_results: list[tuple[str, float]], rerank_scores: dict) -> list:
    """独立调用：位置感知加权融合"""
    blended = []
    for pos, (doc, rrf_score) in enumerate(rrf_results, start=1):
        rr_score = rerank_scores[doc]
        if 1 <= pos <= 3:
            w = config.BLEND_WEIGHTS["top1-3"]
        elif 4 <= pos <= 10:
            w = config.BLEND_WEIGHTS["top4-10"]
        else:
            w = config.BLEND_WEIGHTS["top11+"]
        final_score = rrf_score * w["retrieval"] + rr_score * w["reranker"]
        blended.append((doc, final_score))
    return sorted(blended, key=lambda x: x[1], reverse=True)

def jiajia_search_pipeline(
    query: str,
    documents: list[str],
    query_weight: float = config.DEFAULT_QUERY_WEIGHT,
    bonus_rank1: float = config.DEFAULT_BONUS_RANK1,
    bonus_rank2_3: float = config.DEFAULT_BONUS_RANK2_3
):
    """JiaJia-Search 完整检索重排流水线（一键调用）"""
    print("=" * 80)
    print("🚀 JiaJia-Search 并行混合检索")
    bm25_ranked, vector_ranked = parallel_hybrid_retrieval(query, documents)

    print("🔗 RRF 融合 + Top 排名奖励")
    rrf_top30 = rrf_fusion([bm25_ranked, vector_ranked], query_weight, bonus_rank1, bonus_rank2_3)

    print("🔄 CrossEncoder 重排 + 归一化")
    docs = [d[0] for d in rrf_top30]
    rerank_norm_scores, rel_levels = rerank_with_normalize(query, docs)

    print("⚖️ 位置感知加权融合")
    final_ranked = position_aware_blend(rrf_top30, rerank_norm_scores)

    # 输出结果
    print("\n" + "=" * 100)
    print("🏆 JiaJia-Search 最终排序结果")
    print("=" * 100)
    print(f"{'排名':<5}{'最终得分':<10}{'重排分':<12}{'相关性等级':<25}{'文档'}")
    print("-" * 100)
    for i, (doc, score) in enumerate(final_ranked, 1):
        print(f"Top{i:<4}{score:<10.4f}{rerank_norm_scores[doc]:<12.4f}{rel_levels[doc]:<24}{doc}")
    print("=" * 100)
    return final_ranked