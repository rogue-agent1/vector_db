#!/usr/bin/env python3
"""Vector database — in-memory similarity search with cosine/euclidean distance."""
import sys, json, math, random

class VectorDB:
    def __init__(self, dim=None):
        self.dim = dim; self.vectors = {}; self.metadata = {}
    def _cosine(self, a, b):
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(x*x for x in b))
        return dot / (na * nb) if na and nb else 0
    def _euclidean(self, a, b):
        return -math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))
    def insert(self, id, vector, meta=None):
        if self.dim is None: self.dim = len(vector)
        assert len(vector) == self.dim, f"Expected dim {self.dim}, got {len(vector)}"
        self.vectors[id] = vector
        if meta: self.metadata[id] = meta
    def search(self, query, k=5, metric="cosine"):
        fn = self._cosine if metric == "cosine" else self._euclidean
        scores = sorted(((id, fn(query, v)) for id, v in self.vectors.items()),
                       key=lambda x: -x[1])
        return [{"id": id, "score": round(s, 4), "meta": self.metadata.get(id)}
                for id, s in scores[:k]]

if __name__ == "__main__":
    db = VectorDB(dim=32)
    random.seed(42)
    categories = ["science", "art", "tech", "sports", "music"]
    for i in range(200):
        cat = categories[i % len(categories)]
        base = [0.0] * 32
        base[hash(cat) % 32] = 1.0  # category signal
        vec = [base[j] + random.gauss(0, 0.3) for j in range(32)]
        db.insert(f"doc-{i}", vec, {"category": cat, "id": i})
    query = db.vectors["doc-0"]  # find similar to doc-0 (science)
    results = db.search(query, k=8)
    print(f"VectorDB: {len(db.vectors)} vectors, dim={db.dim}")
    print(f"\nNearest to doc-0 (science):")
    for r in results:
        print(f"  {r['id']:>8s}  score={r['score']:.4f}  {r['meta']}")
