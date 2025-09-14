from functools import lru_cache
from typing import List

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel:
    """Handles embedding generation using HuggingFace transformer models."""

    def __init__(self, model_name: str = "thenlper/gte-small", device: str = "cuda"):
        """
        Initialize the embedding model with the specified pre-trained model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda', 'cpu', etc.)
        """
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        # Add embedding cache
        self._query_cache = {}

    @lru_cache(maxsize=1024)
    def embed_query(self, query: str) -> Tensor:
        """
        Generate embeddings for a given text query.

        Args:
            query: Text to embed

        Returns:
            Tensor containing the embedding
        """
        batch_dict = self.tokenizer(
            query, max_length=512, padding=True, truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        embeddings = average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )
        return embeddings

    def embed_queries(self, queries: List[str]) -> List[Tensor]:
        """
        Batch process multiple queries for more efficient embedding.

        Args:
            queries: List of text queries to embed

        Returns:
            List of tensor embeddings
        """
        results = []
        for query in queries:
            results.append(self.embed_query(query))
        return results

    def to_list(self, embedding: Tensor) -> List[float]:
        """Convert a tensor embedding to a Python list."""
        return embedding.squeeze().cpu().tolist()


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Perform average pooling on the token embeddings, masking out padding tokens.

    Args:
        last_hidden_states: Output embeddings from the model
        attention_mask: Attention mask identifying padding tokens

    Returns:
        Pooled embedding tensor
    """
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def warm_up_embedding_model(
    model_name: str = None, device: str = "cuda"
) -> EmbeddingModel:
    """
    Warm up the embedding model by running a dummy query.

    Args:
        model_name: HuggingFace model identifier
        device: Device to run model on ('cuda', 'cpu', etc.)

    Returns:
        Initialized EmbeddingModel instance
    """
    embedding_model = EmbeddingModel(model_name=model_name, device=device)
    # Run a dummy query to warm up the model
    _ = embedding_model.embed_query("what is the price of tesla")
    return embedding_model


if __name__ == "__main__":
    # Example usage
    embedding_model = EmbeddingModel(
        model_name="Omartificial-Intelligence-Space/GATE-AraBert-v1", device="cuda"
    )

    # Test single query embedding
    query = "what is the price of tesla?"
    embedding = embedding_model.embed_query(query)
    print(f"Type: {type(embedding)}\nShape: {embedding.shape}")

    # Convert to list
    embedding_list = embedding_model.to_list(embedding)
    print(f"Type: {type(embedding_list)}\nLength: {len(embedding_list)}")
