from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from shared.core.logger import get_logger
from shared.core.vector_store import get_llm
from chatbot.retriever import retrieve_policy_chunks
from chatbot.schemas import CitedAnswerOutput

logger = get_logger(__name__)

SYSTEM_PROMPT = """당신은 사내 규정을 안내하는 AI 어시스턴트입니다.
반드시 아래 제공된 [참고 규정 문맥]에 포함된 사실만을 근거로 답변하세요.

[작성 규칙]
1. 문맥에 근거가 전혀 없다면 추측하지 말고 answer에 "제공된 사내 규정에서 관련 내용을 찾을 수 없습니다."라고 작성하고 cited_numbers는 빈 리스트([])로 반환하세요.
2. 답변 문장 끝에는 반드시 근거가 된 문맥 번호(예: [1], [2])를 표기하세요.
3. 실제 인용한 문맥 번호만 cited_numbers에 정수로 포함하세요.

[참고 규정 문맥]
{context}
"""

def _format_numbered_context(docs: list[Document]) -> str:
    blocks = []
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata
        header = f"[{idx}] 문서ID: {meta.get('doc_id')} | 부서: {meta.get('department')} | 페이지: {meta.get('page')}p"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)

def ask_policy_chatbot(
    question: str,
    department: str | None = None,
    top_k: int = 4,
    score_threshold: float = 0.35,
) -> dict:
    """검색 -> 문맥 주입 -> LLM 모델 구조화 생성 -> 출처 후처리를 수행합니다."""
    docs = retrieve_policy_chunks(question, department, top_k, score_threshold)
    if not docs:
        return {
            "question": question,
            "answer": "제공된 사내 규정에서 관련 내용을 찾을 수 없습니다.",
            "citations": [],
            "retrieved_count": 0,
        }

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])
    structured_llm = get_llm().with_structured_output(CitedAnswerOutput)
    chain = prompt | structured_llm

    result: CitedAnswerOutput = chain.invoke({
        "context": _format_numbered_context(docs),
        "question": question,
    })

    # 후처리 - LLM이 실제 인용했다고 밝힌 번호(cited_numbers)에 해당하는 문서 메타데이터만 매핑
    citations = []
    for num in result.cited_numbers:
        if 1 <= num <= len(docs):
            meta = docs[num - 1].metadata
            citations.append({
                "ref_number": num,
                "doc_id": meta.get("doc_id"),
                "department": meta.get("department"),
                "page": meta.get("page"),
            })

    return {
        "question": question,
        "answer": result.answer,
        "citations": citations,
        "retrieved_count": len(docs),
    }