"""
llm_eval_project/src/rag_pipeline.py
──────────────────────────────────────────────────────────────────────────────
Lightweight RAG Pipeline
Bharadwaj Rachuri | github.com/br23aay
──────────────────────────────────────────────────────────────────────────────

A self-contained Retrieval-Augmented Generation pipeline that works
entirely locally — no API keys, no external services required.

Pipeline stages:
  1. INGEST   — load text documents
  2. CHUNK    — split into overlapping chunks
  3. INDEX    — TF-IDF based vector index (no GPU needed)
  4. RETRIEVE — cosine similarity retrieval for a query
  5. GENERATE — prompt construction with retrieved context
  6. EVALUATE — pass results to LLMEvaluator

The generate() step returns a prompt-ready string that can be sent to
any LLM (OpenAI, Azure AI, local Ollama, etc.) or used with the
mock generator included here for demonstration.
"""

from __future__ import annotations

import re
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ─── DOCUMENT ────────────────────────────────────────────────────────────────

@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Document(id={self.doc_id!r}, chars={len(self.text)})"


# ─── CHUNK ───────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)


def _split_into_chunks(
    doc: Document,
    chunk_size: int = 300,
    overlap: int = 60,
) -> list[Chunk]:
    """
    Split a document into overlapping text chunks.

    Parameters
    ----------
    chunk_size : target characters per chunk
    overlap    : character overlap between consecutive chunks
    """
    text = doc.text
    chunks: list[Chunk] = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at sentence boundary
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                chunk_id=f"{doc.doc_id}_chunk_{chunk_idx:04d}",
                doc_id=doc.doc_id,
                text=chunk_text,
                start_char=start,
                end_char=end,
                metadata=doc.metadata.copy(),
            ))
            chunk_idx += 1

        start = end - overlap if end < len(text) else len(text)

    return chunks


# ─── TF-IDF INDEX ────────────────────────────────────────────────────────────

_STOP_WORDS = {
    "the", "a", "an", "is", "it", "in", "of", "to", "for", "and", "or",
    "but", "on", "at", "by", "as", "be", "was", "are", "were", "has",
    "have", "had", "with", "from", "that", "this", "which", "not", "no",
}


def _tokenise(text: str) -> list[str]:
    return [
        w.lower() for w in re.findall(r"\w+", text)
        if w.lower() not in _STOP_WORDS and len(w) > 2
    ]


def _cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vec_a)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


class TFIDFIndex:
    """
    Minimal TF-IDF index over text chunks.
    No dependencies beyond the standard library.
    """

    def __init__(self):
        self._chunks: list[Chunk] = []
        self._tfidf_vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self._chunks.extend(chunks)
        self._rebuild()

    def _rebuild(self) -> None:
        n = len(self._chunks)
        if n == 0:
            return

        # Term frequency per chunk
        tf_lists = []
        df: dict[str, int] = defaultdict(int)

        for chunk in self._chunks:
            tokens = _tokenise(chunk.text)
            tf: dict[str, float] = defaultdict(float)
            for tok in tokens:
                tf[tok] += 1.0
            # Normalise TF
            max_tf = max(tf.values()) if tf else 1
            tf = {t: v / max_tf for t, v in tf.items()}
            tf_lists.append(tf)
            for t in tf:
                df[t] += 1

        # IDF
        self._idf = {t: math.log((n + 1) / (cnt + 1)) + 1 for t, cnt in df.items()}

        # TF-IDF vectors
        self._tfidf_vectors = [
            {t: tf_val * self._idf.get(t, 1.0) for t, tf_val in tf.items()}
            for tf in tf_lists
        ]

    def search(self, query: str, top_k: int = 3) -> list[tuple[Chunk, float]]:
        """Return top_k chunks most similar to query."""
        if not self._chunks:
            return []

        q_tokens = _tokenise(query)
        q_vec: dict[str, float] = defaultdict(float)
        for tok in q_tokens:
            q_vec[tok] += self._idf.get(tok, 1.0)

        scores = [
            (chunk, _cosine_similarity(q_vec, vec))
            for chunk, vec in zip(self._chunks, self._tfidf_vectors)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ─── RAG PIPELINE ────────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    query: str
    retrieved_chunks: list[Chunk]
    retrieval_scores: list[float]
    context: str
    prompt: str
    generated_output: str = ""


class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Usage
    -----
    >>> pipeline = RAGPipeline(chunk_size=300, overlap=60, top_k=3)
    >>> pipeline.ingest([Document("doc1", "Your document text here...")])
    >>> result = pipeline.query("What is the capital of France?")
    >>> print(result.context)
    """

    def __init__(
        self,
        chunk_size: int = 300,
        overlap: int = 60,
        top_k: int = 3,
        context_max_chars: int = 900,
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.context_max_chars = context_max_chars
        self._index = TFIDFIndex()
        self._documents: list[Document] = []
        self._total_chunks = 0

    def ingest(self, documents: list[Document]) -> None:
        """
        Ingest a list of documents — chunk and index them.
        """
        all_chunks: list[Chunk] = []
        for doc in documents:
            chunks = _split_into_chunks(doc, self.chunk_size, self.overlap)
            all_chunks.extend(chunks)
            self._documents.append(doc)

        self._index.add_chunks(all_chunks)
        self._total_chunks += len(all_chunks)
        print(
            f"  Ingested {len(documents)} document(s) → "
            f"{self._total_chunks} total chunks indexed"
        )

    def retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        """Retrieve top_k chunks for a query."""
        return self._index.search(query, top_k=self.top_k)

    def build_prompt(self, query: str, context: str) -> str:
        """Construct a grounded RAG prompt."""
        return (
            f"You are a helpful assistant. Answer the question using ONLY the "
            f"information provided in the context below. "
            f"If the answer is not in the context, say so clearly.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
        )

    def query(
        self,
        query: str,
        generator_fn: Optional[callable] = None,
    ) -> RAGResult:
        """
        Run the full RAG pipeline for a single query.

        Parameters
        ----------
        query        : user question
        generator_fn : callable(prompt: str) -> str
                       If None, uses the built-in mock generator.

        Returns
        -------
        RAGResult with context, prompt, and generated output
        """
        hits = self.retrieve(query)
        chunks = [c for c, _ in hits]
        scores = [s for _, s in hits]

        # Build context from retrieved chunks (truncated)
        context_parts = [c.text for c in chunks]
        context = "\n\n---\n\n".join(context_parts)
        if len(context) > self.context_max_chars:
            context = context[:self.context_max_chars] + "..."

        prompt = self.build_prompt(query, context)

        if generator_fn is None:
            generator_fn = _mock_generator

        output = generator_fn(prompt)

        return RAGResult(
            query=query,
            retrieved_chunks=chunks,
            retrieval_scores=scores,
            context=context,
            prompt=prompt,
            generated_output=output,
        )

    @property
    def stats(self) -> dict:
        return {
            "documents": len(self._documents),
            "chunks": self._total_chunks,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "top_k": self.top_k,
        }


# ─── MOCK GENERATOR ──────────────────────────────────────────────────────────

def _mock_generator(prompt: str) -> str:
    """
    A deterministic mock LLM that extracts key sentences from the context.
    Used for local demonstration without an API key.
    """
    context_start = prompt.find("CONTEXT:") + 8
    context_end = prompt.find("QUESTION:")
    context = prompt[context_start:context_end].strip()

    question_start = prompt.find("QUESTION:") + 9
    answer_start = prompt.find("ANSWER:") + 7
    question = prompt[question_start:answer_start - 7].strip()

    if not context:
        return "I cannot answer this question as no context was provided."

    # Find the most relevant sentence in context
    sentences = [s.strip() for s in re.split(r"[.!?]", context) if len(s.strip()) > 20]
    if not sentences:
        return "Based on the provided context, I was unable to find a clear answer."

    q_words = set(_tokenise(question))
    best_sentence = max(
        sentences,
        key=lambda s: len(q_words & set(_tokenise(s))),
        default=sentences[0],
    )

    return (
        f"Based on the provided context: {best_sentence.strip()}. "
        f"This information is drawn directly from the indexed documents."
    )
