import json
import re
from pathlib import Path
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(
        self,
        chunks_path: str = "data/chunks.json",
        embeddings_path: str = "data/embeddings.npy",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.model = SentenceTransformer(model_name)
        self.embeddings_path = Path(embeddings_path)
        self.embeddings = self._load_or_build_embeddings()

        self.base_chunks = list(self.chunks)
        self.base_embeddings = self.embeddings.copy()

    def _load_or_build_embeddings(self) -> np.ndarray:
        if self.embeddings_path.exists():
            cached = np.load(self.embeddings_path)
            if len(cached) == len(self.chunks):
                print(f"Loaded cached embeddings from {self.embeddings_path}")
                return cached

        print("Building dense embeddings (this runs once and will be cached)...")
        texts = [c["text"] for c in self.chunks]
        emb = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        emb = np.asarray(emb, dtype=np.float32)
        self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, emb)
        print(f"Saved embeddings to {self.embeddings_path}")
        return emb

    def add_chunks(self, new_chunks: list[dict]) -> int:
        if not new_chunks:
            return 0
        texts = [c["text"] for c in new_chunks]
        new_emb = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        new_emb = np.asarray(new_emb, dtype=np.float32)

        self.chunks.extend(new_chunks)
        self.embeddings = np.vstack([self.embeddings, new_emb])
        self.tokenized_corpus.extend([tokenize(t) for t in texts])
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        return len(new_chunks)

    def reset_to_base(self) -> None:
        self.chunks = list(self.base_chunks)
        self.embeddings = self.base_embeddings.copy()
        self.tokenized_corpus = [tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(
        self,
        query: str,
        course: str | None = None,
        top_k: int = 5,
        mode: str = "hybrid",
        rrf_k: int = 60,
    ) -> list[dict]:
        if course and course != "All":
            candidate_indices = [
                i for i, c in enumerate(self.chunks) if c["course"] == course
            ]
        else:
            candidate_indices = list(range(len(self.chunks)))

        if not candidate_indices:
            return []

        query_tokens = tokenize(query)
        all_bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_ranked = sorted(
            candidate_indices, key=lambda i: all_bm25_scores[i], reverse=True
        )

        query_vec = self.model.encode([query], normalize_embeddings=True)[0]
        cand_matrix = self.embeddings[candidate_indices]
        vec_scores_sub = cand_matrix @ query_vec
        vec_score_map = {
            idx: float(score) for idx, score in zip(candidate_indices, vec_scores_sub)
        }
        vec_ranked = sorted(
            candidate_indices, key=lambda i: vec_score_map[i], reverse=True
        )

        if mode == "bm25":
            final_indices = bm25_ranked[:top_k]
            score_map = {i: float(all_bm25_scores[i]) for i in final_indices}
        elif mode == "vector":
            final_indices = vec_ranked[:top_k]
            score_map = {i: vec_score_map[i] for i in final_indices}
        else:
            rrf_scores = {i: 0.0 for i in candidate_indices}
            for rank, idx in enumerate(bm25_ranked, start=1):
                rrf_scores[idx] += 1.0 / (rrf_k + rank)
            for rank, idx in enumerate(vec_ranked, start=1):
                rrf_scores[idx] += 1.0 / (rrf_k + rank)
            final_indices = sorted(
                candidate_indices, key=lambda i: rrf_scores[i], reverse=True
            )[:top_k]
            score_map = {i: rrf_scores[i] for i in final_indices}

        results = []
        for idx in final_indices:
            item = dict(self.chunks[idx])
            item["score"] = round(score_map[idx], 4)
            results.append(item)
        return results