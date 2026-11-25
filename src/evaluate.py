import json
import re
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
from retriever import HybridRetriever
from pipeline import StarMatePipeline


IN_DOMAIN_BENCHMARK = [
    {
        "query": "How does Gaussian elimination work?",
        "course": "MIT 18.06 Linear Algebra",
        "keywords": ["elimination", "pivot", "matrix"],
    },
    {
        "query": "What is LU factorization?",
        "course": "MIT 18.06 Linear Algebra",
        "keywords": ["lu", "lower", "upper", "factorization"],
    },
    {
        "query": "How does Gauss-Jordan elimination find a matrix inverse?",
        "course": "MIT 18.06 Linear Algebra",
        "keywords": ["inverse", "gauss-jordan", "identity"],
    },
    {
        "query": "What is the column space of a matrix?",
        "course": "MIT 18.06 Linear Algebra",
        "keywords": ["column", "space", "subspace", "combination"],
    },
    {
        "query": "What is Mendel's law of segregation?",
        "course": "MIT 7.03 Genetics",
        "keywords": ["mendel", "allele", "segregation", "gamete"],
    },
    {
        "query": "How does DNA replication work?",
        "course": "MIT 7.03 Genetics",
        "keywords": ["dna", "replication", "strand", "polymerase"],
    },
    {
        "query": "What is genetic linkage and recombination?",
        "course": "MIT 7.03 Genetics",
        "keywords": ["linkage", "recombination", "chromosome", "crossover"],
    },
    {
        "query": "How do mutations affect genes or proteins?",
        "course": "MIT 7.03 Genetics",
        "keywords": ["mutation", "gene", "protein", "phenotype"],
    },
    {
        "query": "What is gradient clipping and why is it used?",
        "course": "Stanford CS229 Machine Learning",
        "keywords": ["gradient", "clipping", "threshold", "norm"],
    },
    {
        "query": "How does backpropagation compute gradients in neural networks?",
        "course": "Stanford CS229 Machine Learning",
        "keywords": ["backpropagation", "gradient", "chain", "layer"],
    },
    {
        "query": "What are word vectors or word embeddings?",
        "course": "Stanford CS229 Machine Learning",
        "keywords": ["word", "vector", "embedding", "softmax"],
    },
    {
        "query": "How do recurrent neural networks handle vanishing gradients?",
        "course": "Stanford CS229 Machine Learning",
        "keywords": ["rnn", "recurrent", "vanishing", "gradient"],
    },
]

OUT_OF_SCOPE_QUERIES = [
    "Who won the 2022 FIFA World Cup?",
    "Give me a recipe for French onion soup.",
    "What was Apple's closing stock price yesterday?",
]


def chunk_matches_keywords(text: str, keywords: list[str]) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in keywords)


def evaluate_retrieval(retriever: HybridRetriever, top_k: int = 5) -> list[dict]:
    summary = []
    configs = [
        ("bm25", False),
        ("vector", False),
        ("hybrid", False),
        ("hybrid", True),
    ]

    for mode, use_course_filter in configs:
        course_precisions = []
        hit_rates = []
        mrrs = []

        for item in IN_DOMAIN_BENCHMARK:
            course_arg = item["course"] if use_course_filter else "All"
            hits = retriever.search(
                query=item["query"],
                course=course_arg,
                top_k=top_k,
                mode=mode,
            )
            if not hits:
                course_precisions.append(0.0)
                hit_rates.append(0.0)
                mrrs.append(0.0)
                continue

            c_prec = sum(1 for h in hits if h["course"] == item["course"]) / len(hits)
            course_precisions.append(c_prec)

            first_relevant_rank = None
            for rank, h in enumerate(hits, start=1):
                if h["course"] == item["course"] and chunk_matches_keywords(
                    h["text"], item["keywords"]
                ):
                    first_relevant_rank = rank
                    break

            hit_rates.append(1.0 if first_relevant_rank is not None else 0.0)
            mrrs.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

        label = f"{mode.upper()} + Course Filter" if use_course_filter else f"{mode.upper()} (No Filter)"
        summary.append(
            {
                "setting": label,
                "course_precision_at_5": round(sum(course_precisions) / len(course_precisions), 3),
                "hit_rate_at_5": round(sum(hit_rates) / len(hit_rates), 3),
                "mrr_at_5": round(sum(mrrs) / len(mrrs), 3),
            }
        )
    return summary


def evaluate_generation(pipeline: StarMatePipeline) -> dict:
    sample_in_domain = [
        IN_DOMAIN_BENCHMARK[0],
        IN_DOMAIN_BENCHMARK[1],
        IN_DOMAIN_BENCHMARK[4],
        IN_DOMAIN_BENCHMARK[5],
        IN_DOMAIN_BENCHMARK[8],
        IN_DOMAIN_BENCHMARK[9],
    ]

    cited_count = 0
    latencies = []
    in_domain_records = []

    for item in sample_in_domain:
        t0 = time.perf_counter()
        res = pipeline.answer(
            query=item["query"],
            course=item["course"],
            task_mode="qa",
            retrieval_mode="hybrid",
            top_k=5,
            language="en",
        )
        elapsed = round(time.perf_counter() - t0, 2)
        latencies.append(elapsed)

        has_citation = bool(re.search(r"\[Source:\s*[^,\]]+,\s*p\.\d+\]", res["answer"]))
        if has_citation:
            cited_count += 1
        in_domain_records.append(
            {
                "query": item["query"],
                "course": item["course"],
                "has_citation": has_citation,
                "latency_sec": elapsed,
            }
        )

    refusal_patterns = [
        "do not cover",
        "does not cover",
        "do not contain",
        "does not contain",
        "not covered",
        "not mentioned",
        "no relevant",
        "cannot answer",
    ]
    refused_count = 0
    oos_records = []

    for q in OUT_OF_SCOPE_QUERIES:
        t0 = time.perf_counter()
        res = pipeline.answer(
            query=q,
            course="All",
            task_mode="qa",
            retrieval_mode="hybrid",
            top_k=5,
            language="en",
        )
        elapsed = round(time.perf_counter() - t0, 2)
        lower_ans = res["answer"].lower()
        refused = any(p in lower_ans for p in refusal_patterns)
        if refused:
            refused_count += 1
        oos_records.append(
            {
                "query": q,
                "properly_refused": refused,
                "latency_sec": elapsed,
                "answer_preview": res["answer"][:160],
            }
        )

    return {
        "in_domain_tested": len(sample_in_domain),
        "citation_compliance_rate": round(cited_count / len(sample_in_domain), 3),
        "avg_latency_sec": round(sum(latencies) / len(latencies), 2),
        "out_of_scope_tested": len(OUT_OF_SCOPE_QUERIES),
        "hallucination_refusal_rate": round(refused_count / len(OUT_OF_SCOPE_QUERIES), 3),
        "in_domain_details": in_domain_records,
        "out_of_scope_details": oos_records,
    }


if __name__ == "__main__":
    print("Running Retrieval Ablation (12 queries across 3 courses)...")
    retriever = HybridRetriever()
    ret_results = evaluate_retrieval(retriever, top_k=5)

    print("\n=== Table 1: Retrieval Ablation Results (Top-5) ===")
    print("| Method | Course Precision@5 | Hit Rate@5 | MRR@5 |")
    print("| :--- | :---: | :---: | :---: |")
    for row in ret_results:
        print(
            f"| {row['setting']} | {row['course_precision_at_5']:.3f} | "
            f"{row['hit_rate_at_5']:.3f} | {row['mrr_at_5']:.3f} |"
        )

    print("\nRunning End-to-End Generation & Refusal Evaluation (9 API calls)...")
    pipeline = StarMatePipeline()
    gen_results = evaluate_generation(pipeline)

    print("\n=== Table 2: End-to-End RAG Generation Evaluation ===")
    print(f"- In-Domain Citation Compliance Rate: {gen_results['citation_compliance_rate'] * 100:.1f}% ({gen_results['in_domain_tested']} queries)")
    print(f"- Out-of-Scope Hallucination Refusal Rate: {gen_results['hallucination_refusal_rate'] * 100:.1f}% ({gen_results['out_of_scope_tested']} queries)")
    print(f"- Average End-to-End Latency: {gen_results['avg_latency_sec']} sec")

    output_path = Path("data/eval_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"retrieval_ablation": ret_results, "generation_eval": gen_results},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nSaved full evaluation report to {output_path}")