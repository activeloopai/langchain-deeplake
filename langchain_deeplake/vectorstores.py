"""Deeplake vector store."""

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any, Callable, Iterable, List, Optional, Tuple, Type, TypeVar
from uuid import uuid4

import deeplake
import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

VST = TypeVar("VST", bound="DeeplakeVectorStore")


class DeeplakeVectorStore(VectorStore):
    """Deeplake vector store integration.

    Setup:
        Install ``langchain-deeplake`` package:

        .. code-block:: bash

            pip install -U langchain-deeplake

    Args:
        dataset_path: Path/URL to store the dataset
        embedding_function: Embedding model to use
        token: Optional Activeloop token
        read_only: Whether to open dataset in read-only mode
        creds: Optional cloud credentials for dataset access
        overwrite: Whether to overwrite existing dataset
    """

    def __init__(
        self,
        dataset_path: str,
        embedding_function: Optional[Embeddings] = None,
        token: Optional[str] = None,
        read_only: bool = False,
        creds: Optional[dict] = None,
        overwrite: bool = False,
    ) -> None:
        """Initialize Deeplake vector store."""
        self.embedding_function = embedding_function
        self.dataset_path = dataset_path
        self.token = token
        self.creds = creds

        exists = deeplake.exists(dataset_path, token=token, creds=creds)

        if overwrite and exists:
            deeplake.delete(dataset_path)

        if exists:
            self.dataset = (
                deeplake.open(dataset_path, token=token, creds=creds)
                if not read_only
                else deeplake.open_read_only(dataset_path, token=token, creds=creds)
            )
        else:
            self.dataset = deeplake.create(
                dataset_path,
                token=token,
                creds=creds,
                schema={
                    "ids": deeplake.types.Text(),
                    "embeddings": deeplake.types.Embedding(),
                    "metadatas": deeplake.types.Dict(),
                    "documents": deeplake.types.Text(),
                },
            )

    def __len__(self) -> int:
        """Return the number of documents in the vector store."""
        return len(self.dataset)

    def get_by_ids(self, ids: List[str], **kwargs: Any) -> List[Document]:
        """Return documents by ID."""
        ids_str = ", ".join([f"'{i}'" for i in ids])
        results = self.dataset.query(f"SELECT * WHERE ids IN ({ids_str})")
        docs = results["documents"][:]
        metadatas = results["metadatas"][:]
        return [
            Document(page_content=docs[i], metadata=metadatas[i].to_dict())
            for i in range(len(results))
        ]

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts to the vector store."""
        # Convert iterator to list
        texts = list(texts)

        # Generate embeddings
        embeddings = self.embedding_function.embed_documents(texts)

        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in texts]

        # Handle metadata
        if metadatas is None:
            metadatas = [{} for _ in texts]

        # Add to dataset
        self.dataset.append(
            {
                "ids": ids,
                "embeddings": embeddings,
                "metadatas": metadatas,
                "documents": texts,
            }
        )
        self.dataset.commit()

        return ids

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Asynchronously add texts to the vector store."""
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.add_texts, **kwargs), texts, metadatas, ids
        )

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        """Delete documents by ID from the vector store."""
        if not ids:
            return False

        ids_str = ", ".join([f"'{i}'" for i in ids])
        # Query to find indices to delete
        query = (
            f"SELECT * FROM (SELECT *, ROW_NUMBER() as row_id) WHERE ids IN ({ids_str})"
        )
        results = self.dataset.query(query)

        if len(results) == 0:
            return False

        # Delete found documents
        for idx in sorted(results.row_ids, reverse=True):
            self.dataset.delete(idx)

        self.dataset.commit()
        return True

    async def adelete(
        self, ids: Optional[List[str]] = None, **kwargs: Any
    ) -> Optional[bool]:
        """Asynchronously delete documents by ID."""
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self.delete, **kwargs), ids
        )

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        """Return documents most similar to query."""
        docs_with_scores = self.similarity_search_with_score(query, k, filter, **kwargs)
        return [doc for doc, _ in docs_with_scores]

    async def asimilarity_search(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Document]:
        """Asynchronously return documents most similar to query."""
        func = partial(self.similarity_search, query, k=k, filter=filter, **kwargs)
        return await asyncio.get_running_loop().run_in_executor(None, func)

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Return documents most similar to query with scores."""
        query_embedding = self.embedding_function.embed_query(query)
        emb_str = ", ".join([str(e) for e in query_embedding])

        # Build TQL query
        tql = f"""
        SELECT * FROM (SELECT documents, metadatas, COSINE_SIMILARITY(embeddings, ARRAY[{emb_str}]) as score)
        """
        if filter:
            conditions = [f"metadatas['{k}'] = '{v}'" for k, v in filter.items()]
            tql += f" WHERE {' AND '.join(conditions)}"

        tql += f" ORDER BY score DESC LIMIT {k}"

        results = self.dataset.query(tql)

        docs_with_scores_columnar = {
            "documents": results["documents"][:],
            "metadatas": results["metadatas"][:],
            "score": results["score"][:],
        }

        docs_with_scores = []
        for i in range(len(results)):
            doc = Document(
                page_content=docs_with_scores_columnar["documents"][i],
                metadata=docs_with_scores_columnar["metadatas"][i].to_dict(),
            )
            score = docs_with_scores_columnar["score"][i]
            docs_with_scores.append((doc, score))

        return docs_with_scores

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[dict] = None, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        """Asynchronously return documents most similar to query with scores."""
        func = partial(
            self.similarity_search_with_score, query, k=k, filter=filter, **kwargs
        )
        return await asyncio.get_running_loop().run_in_executor(None, func)

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return documents selected using maximal marginal relevance."""
        query_embedding = self.embedding_function.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            query_embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using maximal marginal relevance by embedding."""
        emb_str = ", ".join([str(e) for e in embedding])
        # Get initial results
        results = self.dataset.query(
            f"""
            SELECT * FROM (SELECT documents, metadatas, embeddings, 
            COSINE_SIMILARITY(embeddings, ARRAY[{emb_str}]) as score)
            ORDER BY score DESC LIMIT {fetch_k}
        """
        )

        if len(results) == 0:
            return []

        # Extract embeddings and convert to numpy
        embeddings = results["embeddings"][:]

        # Calculate MMR
        selected_indices = []
        remaining_indices = list(range(len(embeddings)))

        for _ in range(min(k, len(embeddings))):
            if not remaining_indices:
                break

            # Calculate MMR scores
            if not selected_indices:
                similarities = results["score"][:]
                mmr_scores = similarities
            else:
                similarities = np.array(
                    [results[i]["score"] for i in remaining_indices]
                )
                selected_embeddings = embeddings[selected_indices]
                remaining_embeddings = embeddings[remaining_indices]

                # Calculate diversity penalty
                diversity_scores = np.max(
                    np.dot(remaining_embeddings, selected_embeddings.T), axis=1
                )
                mmr_scores = (
                    lambda_mult * similarities - (1 - lambda_mult) * diversity_scores
                )

            # Select next document
            next_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(next_idx)
            remaining_indices.remove(next_idx)

        # Return selected documents
        return [
            Document(
                page_content=results[i]["documents"], metadata=results[i]["metadatas"]
            )
            for i in selected_indices
        ]

    @classmethod
    def from_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        dataset_path: str = "mem://langchain",
        **kwargs: Any,
    ) -> VST:
        """Create DeeplakeVectorStore from raw texts."""
        store = cls(dataset_path=dataset_path, embedding_function=embedding, **kwargs)
        store.add_texts(texts=texts, metadatas=metadatas)
        return store

    @classmethod
    async def afrom_texts(
        cls: Type[VST],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        dataset_path: str = "mem://langchain",
        **kwargs: Any,
    ) -> VST:
        """Asynchronously create DeeplakeVectorStore from raw texts."""
        store = cls(dataset_path=dataset_path, embedding_function=embedding, **kwargs)
        await store.aadd_texts(texts=texts, metadatas=metadatas)
        return store

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        """Return relevance score function."""
        return lambda x: x  # Identity function since scores are already normalized
