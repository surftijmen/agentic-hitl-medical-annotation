# rag_memory.py
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import os

class RAGMemory:
    def __init__(self, dim=384, index_path="rag.index", meta_path="rag_meta.json"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            with open(meta_path, "r") as f:
                self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.meta = []

    def embed(self, text):
        return self.model.encode([text])[0].astype("float32")

    # ---------- RETRIEVE ----------
    def search(self, text, k=3):
        if len(self.meta) == 0:
            return []

        vec = self.embed(text).reshape(1, -1)
        D, I = self.index.search(vec, k)

        results = []
        for idx in I[0]:
            if idx < len(self.meta):
                results.append(self.meta[idx])

        return results

    # ---------- STORE VALIDATED CASE ----------
    def add_case(self, retrieval_text, full_record):
        vec = self.embed(retrieval_text).reshape(1, -1)
        self.index.add(vec)
        self.meta.append(full_record)

        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.meta, f, indent=2)
