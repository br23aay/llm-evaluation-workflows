"""
tests/test_rag_pipeline.py
------------------------------------------------------------------------------
Unit tests for src/rag_pipeline.py — chunking, TF-IDF indexing, and
retrieval.

These tests verify the pipeline actually retrieves the correct document
for a query (not just that it runs without error), since retrieval
correctness is the core claim of a RAG system.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rag_pipeline import RAGPipeline, Document


def test_ingest_indexes_one_chunk_per_short_document():
    pipeline = RAGPipeline(chunk_size=300, overlap=60, top_k=3)
    pipeline.ingest([
        Document("france", "Paris is the capital city of France."),
        Document("germany", "Berlin is the capital city of Germany."),
    ])
    assert pipeline.stats["documents"] == 2
    assert pipeline.stats["chunks"] == 2


def test_retrieve_returns_the_document_relevant_to_the_query():
    pipeline = RAGPipeline(chunk_size=300, overlap=60, top_k=1)
    pipeline.ingest([
        Document("france", "Paris is the capital city of France. The Eiffel Tower is located there."),
        Document("germany", "Berlin is the capital city of Germany. The Brandenburg Gate is located there."),
    ])

    hits = pipeline.retrieve("What is the capital of France?")

    assert len(hits) == 1
    top_chunk, score = hits[0]
    assert top_chunk.doc_id == "france"
    assert score > 0.0


def test_query_builds_a_grounded_prompt_with_context_and_question():
    pipeline = RAGPipeline(chunk_size=300, overlap=60, top_k=1)
    pipeline.ingest([
        Document("france", "Paris is the capital city of France."),
    ])

    result = pipeline.query("What is the capital of France?")

    assert "CONTEXT:" in result.prompt
    assert "QUESTION: What is the capital of France?" in result.prompt
    assert "Paris" in result.context
    assert result.generated_output != ""


def test_empty_pipeline_returns_no_hits():
    pipeline = RAGPipeline()
    hits = pipeline.retrieve("Any query")
    assert hits == []
