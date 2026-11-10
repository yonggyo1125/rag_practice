import json
from pathlib import Path
from langchain_core.documents import Document
from shared.core.config import settings
from shared.core.exceptions import NotFoundException
from shared.core.logger import get_logger
from shared.core.vector_store import get_raw_db_connection, get_vector_store
from recommend.schemas import KoreanProductMetadata

logger = get_logger(__name__)

def _fetch_existing_product_ids() -> set[str]:
    """PGVector 컬렉션에 이미 저장된 상품 ID(product_id) 집합을 조회합니다."""
    query = """
        SELECT e.cmetadata->>'product_id'
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = %s
    """
    existing: set[str] = set()
    try:
        with get_raw_pg_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (settings.RECOMMEND_COLLECTION,))
                existing = {row[0] for row in cur.fetchall() if row[0]}
    except Exception as exc:
        logger.debug("기존 상품 ID 조회 스킵: %s", exc)
    return existing

def _item_to_document(item: dict) -> Document:
    """korean_products.json의 단일 객체를 LangChain Document로 변환합니다."""
    page_content = (
        f"[상품코드: {item['product_id']} | 모델코드: {item['model_code']}] {item['title']}\n"
        f"브랜드: {item['brand']} | 카테고리: {item['category']} | 가격: {item['price']:,}원\n"
        f"상세스펙: {item['specs']}\n"
        f"상품설명: {item['description']}\n"
        f"구매자리뷰: {item['reviews']}"
    )
    meta = KoreanProductMetadata(
        product_id=item["product_id"],
        model_code=item["model_code"],
        title=item["title"],
        brand=item["brand"],
        category=item["category"],
        price=int(item["price"]),
    )
    return Document(page_content=page_content, metadata=meta.model_dump())

def sync_korean_products(
    file_path: str = "data/korean_products.json",
    batch_size: int = 100,
) -> dict:
    """1,000건의 상품 JSON을 읽어 배치 단위로 PGVector에 증분 적재합니다."""
    path = Path(file_path)
    if not path.exists():
        raise NotFoundException(f"상품 데이터셋 파일을 찾을 수 없습니다: {file_path}")

    raw_items: list[dict] = json.loads(path.read_text(encoding="utf-8"))
    existing_ids = _fetch_existing_product_ids()

    new_docs = [
        _item_to_document(item)
        for item in raw_items
        if item["product_id"] not in existing_ids
    ]
    skipped_count = len(raw_items) - len(new_docs)

    if new_docs:
        store = get_vector_store(settings.RECOMMEND_COLLECTION)
        for start in range(0, len(new_docs), batch_size):
            batch = new_docs[start : start + batch_size]
            batch_ids = [d.metadata["product_id"] for d in batch]
            store.add_documents(batch, ids=batch_ids)
            logger.info(
                "상품 카탈로그 배치 적재 진행: %d / %d건 완료",
                start + len(batch),
                len(new_docs),
            )

    return {
        "file_path": file_path,
        "total_in_json": len(raw_items),
        "newly_inserted_count": len(new_docs),
        "skipped_existing_count": skipped_count,
    }