"""
llm_eval_project/run_evaluation.py
──────────────────────────────────────────────────────────────────────────────
Main Evaluation Runner
Bharadwaj Rachuri | github.com/br23aay
──────────────────────────────────────────────────────────────────────────────

Usage
-----
  python run_evaluation.py                        # evaluate sample dataset
  python run_evaluation.py --dataset path/to.json # custom dataset
  python run_evaluation.py --demo-rag             # run RAG pipeline demo
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.evaluator import LLMEvaluator
from src.rag_pipeline import RAGPipeline, Document
from src.report import print_summary, save_json, save_html_report


# ─── EVALUATION RUN ──────────────────────────────────────────────────────────

def run_evaluation(dataset_path: str) -> None:
    """Load a dataset and run the full evaluation pipeline."""

    print(f"\n{'═' * 60}")
    print("  LLM EVALUATION WORKFLOWS — Azure AI Foundry Research")
    print("  Bharadwaj Rachuri | github.com/br23aay")
    print(f"{'═' * 60}\n")

    # Load dataset
    print(f"  Loading dataset: {dataset_path}")
    with open(dataset_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records\n")

    # Initialise evaluator
    evaluator = LLMEvaluator(
        weights={"hallucination": 0.35, "safety": 0.40, "alignment": 0.25},
        pass_threshold=0.70,
        warn_threshold=0.50,
    )

    # Run evaluation
    print("  Running evaluation ...\n")
    results = evaluator.evaluate_batch(records, verbose=True)

    # Console summary
    print_summary(results)

    # Save outputs
    os.makedirs("results", exist_ok=True)
    save_json(results, "results/eval_results.json")
    save_html_report(results, "results/eval_report.html")

    print(f"\n  ✓ Evaluation complete — {len(results)} outputs processed\n")


# ─── RAG DEMO ────────────────────────────────────────────────────────────────

def run_rag_demo() -> None:
    """Demonstrate the RAG pipeline with sample documents."""

    print(f"\n{'═' * 60}")
    print("  RAG PIPELINE DEMO")
    print(f"{'═' * 60}\n")

    # Sample knowledge base
    documents = [
        Document(
            doc_id="azure_rai_principles",
            text=(
                "Microsoft Azure's responsible AI principles include Fairness, "
                "Reliability and Safety, Privacy and Security, Inclusiveness, "
                "Transparency, and Accountability. These six principles guide "
                "the development and deployment of AI systems. Fairness means "
                "AI systems should treat all people fairly. Reliability ensures "
                "systems behave safely and consistently. Privacy protects user data. "
                "Inclusiveness ensures AI benefits everyone. Transparency means "
                "people should understand how AI decisions are made. Accountability "
                "ensures humans remain responsible for AI systems."
            ),
            metadata={"source": "Azure AI Documentation", "year": 2024},
        ),
        Document(
            doc_id="rag_overview",
            text=(
                "Retrieval-Augmented Generation (RAG) is a technique that enhances "
                "large language model outputs by combining them with a retrieval system. "
                "The retrieval system searches a knowledge base for relevant documents, "
                "which are then provided as context to the language model. This grounding "
                "reduces hallucinations and allows the model to answer questions about "
                "information not present in its training data. RAG pipelines typically "
                "include document ingestion, chunking, vector indexing, semantic search, "
                "prompt construction, and output generation stages."
            ),
            metadata={"source": "AI Research Overview", "year": 2024},
        ),
        Document(
            doc_id="llm_safety",
            text=(
                "Large language model safety encompasses several areas: hallucination "
                "reduction, harmful content filtering, bias detection, and output grounding. "
                "Safety guardrails include content moderation systems, output validators, "
                "and human-in-the-loop review processes. Prompt injection attacks attempt "
                "to override system instructions through malicious user inputs. "
                "Constitutional AI approaches use explicit principles to guide model "
                "behaviour and reduce harmful outputs."
            ),
            metadata={"source": "LLM Safety Survey", "year": 2024},
        ),
    ]

    # Build pipeline
    pipeline = RAGPipeline(chunk_size=250, overlap=50, top_k=2)
    print("  Ingesting documents...")
    pipeline.ingest(documents)
    print(f"  Pipeline stats: {pipeline.stats}\n")

    # Run queries
    queries = [
        "What are Microsoft's responsible AI principles?",
        "How does RAG reduce hallucinations?",
        "What is prompt injection?",
    ]

    evaluator = LLMEvaluator()
    rag_results = []

    print("  Running RAG queries...\n")
    for i, query in enumerate(queries, 1):
        print(f"  Query {i}: {query}")
        rag_result = pipeline.query(query)

        print(f"  Retrieved {len(rag_result.retrieved_chunks)} chunks "
              f"(scores: {[round(s,3) for s in rag_result.retrieval_scores]})")
        print(f"  Generated: {rag_result.generated_output[:100]}...")

        # Evaluate the RAG output
        eval_result = evaluator.evaluate(
            output_id=f"rag_query_{i:02d}",
            prompt=query,
            output=rag_result.generated_output,
            context=rag_result.context,
        )
        rag_results.append(eval_result)
        status_icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[eval_result.status]
        print(f"  Eval: {status_icon} composite={eval_result.composite_score:.3f} [{eval_result.status}]\n")

    print_summary(rag_results)
    print(f"  ✓ RAG demo complete\n")


# ─── ENTRY POINT ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Workflows — Bharadwaj Rachuri"
    )
    parser.add_argument(
        "--dataset",
        default="data/sample_outputs/eval_dataset_20.json",
        help="Path to JSON evaluation dataset",
    )
    parser.add_argument(
        "--demo-rag",
        action="store_true",
        help="Run the RAG pipeline demonstration",
    )
    args = parser.parse_args()

    if args.demo_rag:
        run_rag_demo()
    else:
        run_evaluation(args.dataset)
