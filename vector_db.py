import re

import numpy as np
from langchain_core.tools import tool
import pickle  # Add this import
import requests
from dotenv import load_dotenv
import os
from langchain_google_vertexai import VertexAIEmbeddings

load_dotenv()
response = requests.get(
    "https://storage.googleapis.com/benchmarks-artifacts/travel-db/swiss_faq.md"
)
project_id = os.getenv("PROJECT_ID")

response.raise_for_status()
faq_text = response.text

docs = [{"page_content": txt} for txt in re.split(r"(?=\n##)", faq_text)]

embedding_model = VertexAIEmbeddings(
    model_name="text-multilingual-embedding-002",
    project=project_id,
)


class VectorStoreRetriever:
    def __init__(self, docs: list, vectors: list, client):
        self._arr = np.array(vectors)
        self._docs = docs
        self._client = client

    @classmethod
    def from_docs(cls, docs, client):
        embeddings = client.embed_documents([doc["page_content"] for doc in docs])
        vectors = embeddings
        cls.save_embeddings(vectors)  # Save embeddings to a pickle file
        return cls(docs, vectors, client)

    @staticmethod
    def save_embeddings(vectors):
        with open("embeddings.pkl", "wb") as f:  # Save to a pickle file
            pickle.dump(np.array(vectors), f)

    @classmethod
    def load_embeddings(cls):
        with open("embeddings.pkl", "rb") as f:  # Load from a pickle file
            return pickle.load(f)

    def query(self, query: str, k: int = 5) -> list[dict]:
        if not hasattr(self, "_arr"):  # Check if embeddings are loaded
            self._arr = self.load_embeddings()  # Load embeddings if not already loaded
        embed = self._client.embed_query(query)  # Use embed_query instead of embed
        # "@" is just a matrix multiplication in python
        scores = np.array(embed) @ self._arr.T
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx_sorted = top_k_idx[np.argsort(-scores[top_k_idx])]
        return [
            {**self._docs[idx], "similarity": scores[idx]} for idx in top_k_idx_sorted
        ]


# retriever = VectorStoreRetriever.from_docs(docs, openai.Client())
retriever = VectorStoreRetriever.from_docs(docs, embedding_model)
