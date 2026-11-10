from fastapi import APIRouter
from shared.core.schemas import APIResponse
from chatbot.chunker import preview_chunking
from chatbot.generator import ask_policy_chatbot
from chatbot.indexer import index_policy_document
from chatbot.schemas import (
    ChatbotAskRequest,
    ChunkPreviewRequest,
    IndexDocumentRequest,
)

router = APIRouter(prefix="/v1/chatbot", tags=["[과제 1] 사내 규정 챗봇"])

@router.post("/chunks/preview", response_model=APIResponse[dict])
def preview_chunks_endpoint(req: ChunkPreviewRequest):
    """청킹 사이즈 전략별 분할 결과를 비교합니다."""
    data = preview_chunking(req.file_path, req.small_chunk_size, req.large_chunk_size, req.chunk_overlap)
    return APIResponse.ok(data=data, message="청킹 전략 비교 실험이 완료되었습니다.")

@router.post("/documents/index", response_model=APIResponse[dict])
def index_document_endpoint(req: IndexDocumentRequest):
    """문서를 청킹하고 SHA-256 해시 기반으로 PGVector에 증분 적재합니다."""
    data = index_policy_document(req.file_path, req.doc_id, req.department, req.chunk_size, req.chunk_overlap)
    return APIResponse.ok(data=data, message="사내 규정 문서 증분 인덱싱이 완료되었습니다.")

@router.post("/ask", response_model=APIResponse[dict])
def ask_chatbot_endpoint(req: ChatbotAskRequest):
    """부서 필터링 검색 및 출처 표기 기반 답변을 생성합니다."""
    data = ask_policy_chatbot(req.question, req.department, req.top_k, req.score_threshold)
    return APIResponse.ok(data=data, message="사내 규정 질의응답 생성이 완료되었습니다.")