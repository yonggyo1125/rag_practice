from fastapi import APIRouter
from shared.core.schemas import APIResponse
from recommend.catalog_loader import sync_korean_products
from recommend.hybrid_rrf import hybrid_rrf_search
from recommend.query_rewriter import rewrite_query_from_activity
from recommend.reranker import rerank_and_compress_candidates
from recommend.schemas import (
    CatalogSyncRequest,
    HybridSearchRequest,
    PersonalizedRecommendRequest,
)

router = APIRouter(prefix="/v1/recommend", tags=["[실습 2] 고객 활동 로그 기반 하이브리드 상품 추천"])

@router.post("/catalog/sync", response_model=APIResponse[dict])
def sync_catalog_endpoint(req: CatalogSyncRequest):
    """data/korean_products.json(1,000건)을 배치 단위로 PGVector에 증분 적재합니다."""
    data = sync_korean_products(file_path=req.file_path, batch_size=req.batch_size)
    return APIResponse.ok(data=data, message="한국어 상품 카탈로그(1,000건) 동기화가 완료되었습니다.")

@router.post("/search/hybrid", response_model=APIResponse[dict])
def hybrid_search_endpoint(req: HybridSearchRequest):
    """rank_bm25와 PGVector의 가중치 튜닝 및 RRF 결합 순위를 비교합니다."""
    results = hybrid_rrf_search(
        keyword_query=req.query,
        semantic_query=req.query,
        bm25_weight=req.bm25_weight,
        vector_weight=req.vector_weight,
        rrf_k=req.rrf_k,
        top_k=req.top_k,
        category=req.category,
    )
    formatted = [
        {
            "product_id": r["document"].metadata.get("product_id"),
            "model_code": r["document"].metadata.get("model_code"),
            "title": r["document"].metadata.get("title"),
            "price": r["document"].metadata.get("price"),
            "bm25_rank": r["bm25_rank"],
            "vector_rank": r["vector_rank"],
            "rrf_score": r["rrf_score"],
        }
        for r in results
    ]
    return APIResponse.ok(data={"query": req.query, "results": formatted})

@router.post("/personalized", response_model=APIResponse[dict])
def personalized_recommend_endpoint(req: PersonalizedRecommendRequest):
    """고객 활동 로그 기반 Query 변환 -> 하이브리드 RRF -> Re-ranking 추천을 수행합니다."""
    rewritten = rewrite_query_from_activity(req.user_request, req.activity_log)

    candidates = hybrid_rrf_search(
        keyword_query=rewritten.keyword_query,
        semantic_query=rewritten.semantic_query,
        top_k=req.top_k * 2,
        category=req.activity_log.preferred_category,
        max_budget=req.activity_log.max_budget,
    )

    recommendations = rerank_and_compress_candidates(
        user_request=req.user_request,
        intent_summary=rewritten.intent_summary,
        hybrid_candidates=candidates,
        top_k=req.top_k,
    )

    return APIResponse.ok(
        data={
            "rewritten_queries": rewritten.model_dump(),
            "recommendations": recommendations,
        },
        message="고객 활동 로그 기반 하이브리드 상품 추천이 완료되었습니다.",
    )