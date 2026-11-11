from fastapi import APIRouter
from shared.core.schemas import APIResponse
from evaluation.dataset_builder import load_or_create_golden_dataset
from evaluation.regression_service import (
    evaluate_single_pipeline,
    run_regression_comparison,
)
from evaluation.schemas import RegressionCompareRequest, RunEvaluationRequest

router = APIRouter(prefix="/v1/evaluations", tags=["[실습 3] RAGAS 정량 평가 및 회귀 검증"])

@router.get("/dataset", response_model=APIResponse[list])
def get_golden_dataset_endpoint():
    """[Day 6] korean_products.json 기반 Golden QA 평가 데이터셋 목록을 조회합니다."""
    dataset = [item.model_dump() for item in load_or_create_golden_dataset()]
    return APIResponse.ok(data=dataset, message="Golden QA 평가셋 조회가 완료되었습니다.")

@router.post("/run", response_model=APIResponse[dict])
def run_evaluation_endpoint(req: RunEvaluationRequest):
    """지정된 파이프라인의 RAGAS 4대 지표를 측정하고 실패 원인을 자동 진단합니다."""
    data = evaluate_single_pipeline(req.pipeline_mode, req.score_threshold_for_failure)
    return APIResponse.ok(data=data, message="RAGAS 정량 평가 및 실패 원인 진단이 완료되었습니다.")

@router.post("/regression", response_model=APIResponse[dict])
def run_regression_endpoint(req: RegressionCompareRequest):
    """기본 벡터 검색(Baseline) 대비 하이브리드+리랭킹(Improved)의 회귀 검증 비교 리포트를 반환합니다."""
    data = run_regression_comparison(top_k=req.top_k)
    return APIResponse.ok(data=data, message="파이프라인 개선 전후 회귀 검증 비교가 완료되었습니다.")