# RAG Pipeline Module
# End-to-end retrieval-augmented generation pipeline

from dataclasses import dataclass, field
from typing import List, Optional, Callable
import math
import re


@dataclass
class Document:
      doc_id: str
      text: str
      metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
      chunk_id: str
      doc_id: str
      text: str


@dataclass
class RAGResult:
      query: str
      retrieved_chunks: List[Chunk]
      context: str
      generated_output: str


class TFIDFIndex:
      def __init__(self):
                self.chunks: List[Chunk] = []
                self.idf: dict = {}
                self.tf: List[dict] = []

      def build(self, chunks: List[Chunk]):
                self.chunks = chunks
                N = len(chunks)
                doc_freq = {}
                self.tf = []
                for chunk in chunks:
                              tokens = self._tokenize(chunk.text)
                              tf = {}
                              for t in tokens:
                                                tf[t] = tf.get(t, 0) + 1 / max(len(tokens), 1)
                                            self.tf.append(tf)
                              for t in set(tokens):
                                                doc_freq[t] = doc_freq.get(t, 0) + 1
                                        self.idf = {t: math.log((N + 1) / (df + 1)) for t, df in doc_freq.items()}

            def query(self, text: str, top_k: int = 3) -> List[Chunk]:
                      q_tokens = self._tokenize(text)
                      scores = []
                      for i, tf in enumerate(self.tf):
                                    score = sum(tf.get(t, 0) * self.idf.get(t, 0) for t in q_tokens)
                                    scores.append((score, i))
                                scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:top_k]]

    def _tokenize(self, text: str) -> List[str]:
              return re.findall(r"[a-z0-9]+", text.lower())


class RAGPipeline:
      def __init__(self, chunk_size: int = 300, overlap: int = 60, top_k: int = 3):
                self.chunk_size = chunk_size
        self.overlap = overlap
        self.top_k = top_k
        self.index = TFIDFIndex()
        self._documents: List[Document] = []

    def ingest(self, documents: List[Document]):
              self._documents = documents
        chunks = []
        for doc in documents:
                      words = doc.text.split()
            step = max(self.chunk_size - self.overlap, 1)
            i = 0
            chunk_num = 0
            while i < len(words):
                              chunk_text = " ".join(words[i : i + self.chunk_size])
                              chunks.append(Chunk(
                                  chunk_id=f"{doc.doc_id}_c{chunk_num}",
                                  doc_id=doc.doc_id,
                                  text=chunk_text,
                              ))
                              i += step
                              chunk_num += 1
                      self.index.build(chunks)

    def query(
              self,
              query_text: str,
              generator_fn: Optional[Callable[[str], str]] = None,
    ) -> RAGResult:
              retrieved = self.index.query(query_text, top_k=self.top_k)
        context = " ".join(c.text for c in retrieved)
        if generator_fn:
                      prompt = f"Context: {context}\n\nQuestion: {query_text}\n\nAnswer:"
            output = generator_fn(prompt)
else:
            output = f"[Mock] Based on context, answering: {query_text}"
        return RAGResult(
                      query=query_text,
                      retrieved_chunks=retrieved,
                      context=context,
                      generated_output=output,
        )
