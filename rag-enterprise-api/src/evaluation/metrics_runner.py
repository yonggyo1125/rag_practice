from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from shared.core.vector_store import get_embeddings, get_llm

def run_ragas_metrics(collected_samples: list[dict]) -> list[dict]:
    """RAGAS 4대 핵심 지표를 측정하여 샘플별 점수를 반환합니다."""
    ragas_dataset = Dataset.from_dict({
        "question": [s["question"] for s in collected_samples],
        "contexts": [s["contexts"] for s in collected_samples],
        "answer": [s["answer"] for s in collected_samples],
        "ground_truth": [s["ground_truth"] for s in collected_samples],
    })

    evaluator_llm = LangchainLLMWrapper(get_llm())
    evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]

    eval_result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )

    scores_df = eval_result.to_pandas()
    enriched_results: list[dict] = []

    for idx, sample in enumerate(collected_samples):
        row = scores_df.iloc[idx]
        enriched_results.append({
            **sample,
            "context_precision": round(float(row.get("context_precision", 0.0) or 0.0), 4),
            "context_recall": round(float(row.get("context_recall", 0.0) or 0.0), 4),
            "faithfulness": round(float(row.get("faithfulness", 0.0) or 0.0), 4),
            "answer_relevancy": round(float(row.get("answer_relevancy", 0.0) or 0.0), 4),
        })

    return enriched_results