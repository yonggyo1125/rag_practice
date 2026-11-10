import re
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from shared.core.config import settings
from shared.core.logger import get_logger
from shared.core.vector_store import get_raw_db_connection

logger = get_logger(__name__)

def tokenize_korean_catalog(text: str) -> list[str]:
    """한글 어절, 영문 브랜드, 하이픈/슬래시 모델코드(예: 16zd90sp-kr0193, rtx4070)를 보존하여 토큰화합니다."""
    cleaned = re.sub(r"[^\w\s\-/]", " ", text.lower())
    tokens: list[str] = []
    for raw_token in cleaned.split():
        if len(raw_token) > 1:
            tokens.append(raw_token)
            # 하이픈 결합 모델코드(예: 16zd90sp-kr0193)는 베이스 모델명(16zd90sp)으로도 매칭되도록 보조 토큰 추가
            if "-" in raw_token:
                tokens.extend(part for part in raw_token.split("-") if len(part) > 1)
    return tokens

def fetch_filtered_catalog_docs(
    category: str | None = None,
    max_budget: int | None = None,
) -> list[Document]:
    """PGVector에서 카테고리 및 예산 조건을 만족하는 상품 문서들을 로드합니다."""
    query = """
        SELECT e.document, e.cmetadata
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
    """
    docs: list[Document] = []
    with get_raw_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (settings.RECOMMEND_COLLECTION,))
            for content, meta in cur.fetchall():
                metadata = meta or {}
                if category and metadata.get("category") != category:
                    continue
                if max_budget is not None and metadata.get("price", 0) > max_budget:
                    continue
                docs.append(Document(page_content=content, metadata=metadata))
    return docs

def search_bm25_catalog(
    query: str,
    top_k: int = 8,
    category: str | None = None,
    max_budget: int | None = None,
) -> list[tuple[Document, float]]:
    """rank_bm25를 이용해 모델코드·스펙 키워드 매칭 상위 상품을 검색합니다."""
    docs = fetch_filtered_catalog_docs(category=category, max_budget=max_budget)
    if not docs:
        return []

    tokenized_corpus = [tokenize_korean_catalog(doc.page_content) for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)

    query_tokens = tokenize_korean_catalog(query)
    scores = bm25.get_scores(query_tokens)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [(doc, float(score)) for doc, score in ranked[:top_k]]