from langchain_core.documents import Document
from shared.core.config import settings
from shared.core.vector_store import get_vector_store
from recommend.bm25_retriever import search_bm25_catalog

def search_vector_catalog(
    query: str,
    top_k: int = 8,
    category: str | None = None,
    max_budget: int | None = None,
) -> list[tuple[Document, float]]:
    """PGVector에서 JSONB Pre-filtering(카테고리·예산)과 코사인 유사도 검색을 수행합니다."""
    store = get_vector_store(settings.RECOMMEND_COLLECTION)
    conditions: list[dict] = []
    if category:
        conditions.append({"category": {"$eq": category}})
    if max_budget is not None:
        conditions.append({"price": {"$lte": max_budget}})

    if len(conditions) == 1:
        filter_dict = conditions[0]
    elif len(conditions) > 1:
        filter_dict = {"$and": conditions}
    else:
        filter_dict = None

    return store.similarity_search_with_relevance_scores(query, k=top_k, filter=filter_dict)

def hybrid_rrf_search(
    keyword_query: str,
    semantic_query: str,
    bm25_weight: float = 0.5,
    vector_weight: float = 0.5,
    rrf_k: int = 60,
    top_k: int = 5,
    category: str | None = None,
    max_budget: int | None = None,
) -> list[dict]:
    """BM25 순위와 PGVector 순위를 가중치 기반 RRF 수식으로 결합합니다."""
    pool_size = top_k * 3
    bm25_hits = search_bm25_catalog(keyword_query, top_k=pool_size, category=category, max_budget=max_budget)
    vector_hits = search_vector_catalog(semantic_query, top_k=pool_size, category=category, max_budget=max_budget)

    registry: dict[str, dict] = {}

    for rank, (doc, raw_score) in enumerate(bm25_hits, start=1):
        pid = doc.metadata.get("product_id")
        registry[pid] = {
            "document": doc,
            "bm25_rank": rank,
            "bm25_raw_score": round(raw_score, 4),
            "vector_rank": None,
            "vector_raw_score": 0.0,
            "rrf_score": bm25_weight * (1.0 / (rrf_k + rank)),
        }

    for rank, (doc, raw_score) in enumerate(vector_hits, start=1):
        pid = doc.metadata.get("product_id")
        addition = vector_weight * (1.0 / (rrf_k + rank))
        if pid in registry:
            registry[pid]["vector_rank"] = rank
            registry[pid]["vector_raw_score"] = round(float(raw_score), 4)
            registry[pid]["rrf_score"] += addition
        else:
            registry[pid] = {
                "document": doc,
                "bm25_rank": None,
                "bm25_raw_score": 0.0,
                "vector_rank": rank,
                "vector_raw_score": round(float(raw_score), 4),
                "rrf_score": addition,
            }

    ranked_items = sorted(registry.values(), key=lambda x: x["rrf_score"], reverse=True)
    for item in ranked_items:
        item["rrf_score"] = round(item["rrf_score"], 6)

    return ranked_items[:top_k]