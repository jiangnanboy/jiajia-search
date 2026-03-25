from .config import config
def rrf_fusion(
    retrieval_lists: list[list[tuple[str, int]]],
    query_weight: float = 2.0,
    bonus_rank1: float = 0.05,
    bonus_rank2_3: float = 0.02
) -> list[tuple[str, float]]:
    """独立调用：RRF 多路结果融合 + Top 排名奖励"""
    doc_score_map = {}
    for ranked_list in retrieval_lists:
        for doc, rank in ranked_list:
            rrf_base = 1 / (config.RRF_K + rank + 1)
            rrf_weighted = rrf_base * query_weight
            doc_score_map[doc] = doc_score_map.get(doc, 0.0) + rrf_weighted

    sorted_initial = sorted(doc_score_map.items(), key=lambda x: x[1], reverse=True)
    for idx, (doc, score) in enumerate(sorted_initial, start=1):
        if idx == 1:
            doc_score_map[doc] += bonus_rank1
        elif 2 <= idx <= 3:
            doc_score_map[doc] += bonus_rank2_3

    return sorted(doc_score_map.items(), key=lambda x: x[1], reverse=True)[:config.TOP_K_RECALL]