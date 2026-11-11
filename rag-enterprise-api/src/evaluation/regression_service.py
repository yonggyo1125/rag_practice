from evaluation.dataset_builder import collect_pipeline_outputs
from evaluation.failure_analyzer import diagnose_sample_failure
from evaluation.metrics_runner import run_ragas_metrics

def _summarize_averages(results: list[dict]) -> dict[str, float]:
    count = len(results) or 1
    return {
        "context_precision": round(sum(r["context_precision"] for r in results) / count, 4),
        "context_recall": round(sum(r["context_recall"] for r in results) / count, 4),
        "faithfulness": round(sum(r["faithfulness"] for r in results) / count, 4),
        "answer_relevancy": round(sum(r["answer_relevancy"] for r in results) / count, 4),
    }

def evaluate_single_pipeline(pipeline_mode: str, threshold: float = 0.7) -> dict:
    """단일 파이프라인의 RAGAS 평가 및 실패 원인 진단을 수행합니다."""
    samples = collect_pipeline_outputs(pipeline_mode=pipeline_mode)
    scored = run_ragas_metrics(samples)
    diagnosed = [diagnose_sample_failure(s, threshold=threshold).model_dump() for s in scored]

    return {
        "pipeline_mode": pipeline_mode,
        "average_metrics": _summarize_averages(scored),
        "sample_details": diagnosed,
    }

def run_regression_comparison(top_k: int = 3) -> dict:
    """Baseline(vector_only) 대비 Improved(hybrid_rerank)의 성능 향상을 비교 검증합니다."""
    base_samples = collect_pipeline_outputs("vector_only", top_k=top_k)
    hybrid_samples = collect_pipeline_outputs("hybrid_rerank", top_k=top_k)

    base_scored = run_ragas_metrics(base_samples)
    hybrid_scored = run_ragas_metrics(hybrid_samples)

    base_avg = _summarize_averages(base_scored)
    hybrid_avg = _summarize_averages(hybrid_scored)

    improvement_delta = {
        metric: round(hybrid_avg[metric] - base_avg[metric], 4)
        for metric in base_avg
    }

    return {
        "baseline_vector_only": base_avg,
        "improved_hybrid_rerank": hybrid_avg,
        "improvement_delta": improvement_delta,
        "regression_passed": all(delta >= -0.02 for delta in improvement_delta.values()),
    }