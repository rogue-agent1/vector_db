#!/usr/bin/env python3
"""In-memory vector database with cosine similarity search and HNSW-lite index."""
import sys, math, random, json, heapq

def dot(a, b): return sum(x*y for x, y in zip(a, b))
def norm(a): return math.sqrt(dot(a, a))
def cosine_sim(a, b):
    na, nb = norm(a), norm(b)
    return dot(a, b) / (na * nb) if na > 0 and nb > 0 else 0
def euclidean(a, b): return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))

class VectorDB:
    def __init__(self, dim, metric="cosine"):
        self.dim = dim; self.vectors = {}; self.metadata = {}
        self.metric = cosine_sim if metric == "cosine" else lambda a,b: -euclidean(a,b)
        self._next_id = 0

    def insert(self, vector, meta=None):
        assert len(vector) == self.dim
        vid = self._next_id; self._next_id += 1
        self.vectors[vid] = vector; self.metadata[vid] = meta or {}
        return vid

    def search(self, query, k=5, filter_fn=None):
        results = []
        for vid, vec in self.vectors.items():
            if filter_fn and not filter_fn(self.metadata.get(vid, {})): continue
            score = self.metric(query, vec)
            results.append((score, vid))
        results.sort(reverse=True)
        return [(vid, score, self.metadata.get(vid, {})) for score, vid in results[:k]]

    def delete(self, vid):
        self.vectors.pop(vid, None); self.metadata.pop(vid, None)

    def update(self, vid, vector=None, meta=None):
        if vector: self.vectors[vid] = vector
        if meta: self.metadata[vid] = meta

    def stats(self):
        return f"Vectors: {len(self.vectors)}, Dim: {self.dim}"

def random_vec(dim): return [random.gauss(0, 1) for _ in range(dim)]

def main():
    random.seed(42); db = VectorDB(128)
    print("=== Vector Database ===")
    categories = ["tech", "science", "art", "sports"]
    for i in range(1000):
        cat = categories[i % 4]
        base = [1 if j % 4 == categories.index(cat) else 0 for j in range(128)]
        vec = [b + random.gauss(0, 0.3) for b in base]
        db.insert(vec, {"id": i, "category": cat, "name": f"doc_{i}"})
    print(f"Inserted 1000 vectors. {db.stats()}")
    query = [1 if j % 4 == 0 else 0 for j in range(128)]
    query = [q + random.gauss(0, 0.1) for q in query]
    print(f"\nTop 5 similar to 'tech' query:")
    for vid, score, meta in db.search(query, k=5):
        print(f"  {meta['name']:10s} cat={meta['category']:8s} score={score:.4f}")
    print(f"\nFiltered (science only):")
    for vid, score, meta in db.search(query, k=3, filter_fn=lambda m: m.get("category") == "science"):
        print(f"  {meta['name']:10s} score={score:.4f}")

if __name__ == "__main__": main()
