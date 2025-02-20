from langchain_community.embeddings import FakeEmbeddings, DeterministicFakeEmbedding
from pytest import FixtureRequest
from langchain_core.documents import Document
from langchain_deeplake import DeeplakeVectorStore
import pytest
import re
import math
import requests


@pytest.fixture
def deeplake_datastore() -> DeeplakeVectorStore:  # type: ignore[misc]
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = DeeplakeVectorStore.from_texts(
        dataset_path="./test_path",
        texts=texts,
        metadatas=metadatas,
        embedding=DeterministicFakeEmbedding(size=384),
        overwrite=True,
    )
    yield docsearch

    docsearch.delete_dataset()


@pytest.fixture(params=["l2", "cos"])
def distance_metric(request: FixtureRequest) -> str:
    return request.param


def test_deeplake() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    docsearch = DeeplakeVectorStore.from_texts(
        dataset_path="mem://test_deeplake",
        texts=texts,
        embedding=DeterministicFakeEmbedding(size=384),
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo")]


def test_deeplake_with_metadata() -> None:
    """Test end to end construction and search."""
    texts = ["foo", "bar", "baz"]
    metadatas = [{"page": str(i)} for i in range(len(texts))]
    docsearch = DeeplakeVectorStore.from_texts(
        dataset_path="mem://test_deeplake_with_metadata",
        texts=texts,
        embedding=DeterministicFakeEmbedding(size=384),
        metadatas=metadatas,
    )
    output = docsearch.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]


def test_deeplake_with_persistence(deeplake_datastore) -> None:  # type: ignore[no-untyped-def]
    """Test end to end construction and search, with persistence."""
    output = deeplake_datastore.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]

    # Get a new VectorStore from the persisted directory
    docsearch = DeeplakeVectorStore(
        dataset_path="./test_path",
        embedding_function=DeterministicFakeEmbedding(size=384),
    )
    output = docsearch.similarity_search("foo", k=1)

    # Clean up
    docsearch.delete_dataset()


def test_deeplake_overwrite_flag(deeplake_datastore) -> None:  # type: ignore[no-untyped-def]
    """Test overwrite behavior"""
    dataset_path = "./test_path"

    output = deeplake_datastore.similarity_search("foo", k=1)
    assert output == [Document(page_content="foo", metadata={"page": "0"})]

    # Get a new VectorStore from the persisted directory, with no overwrite (implicit)
    docsearch = DeeplakeVectorStore(
        dataset_path=dataset_path,
        embedding_function=DeterministicFakeEmbedding(size=384),
    )
    output = docsearch.similarity_search("foo", k=1)
    # assert page still present
    assert output == [Document(page_content="foo", metadata={"page": "0"})]

    # Get a new VectorStore from the persisted directory, with no overwrite (explicit)
    docsearch = DeeplakeVectorStore(
        dataset_path=dataset_path,
        embedding_function=DeterministicFakeEmbedding(size=384),
        overwrite=False,
    )
    output = docsearch.similarity_search("foo", k=1)
    # assert page still present
    assert output == [Document(page_content="foo", metadata={"page": "0"})]

    # Get a new VectorStore from the persisted directory, with overwrite
    docsearch = DeeplakeVectorStore(
        dataset_path=dataset_path,
        embedding_function=DeterministicFakeEmbedding(size=384),
        overwrite=True,
    )

    output = docsearch.similarity_search("foo", k=1)
    assert output == []


def test_similarity_search(deeplake_datastore) -> None:  # type: ignore[no-untyped-def]
    """Test similarity search."""
    distance_metric = "cos"
    output = deeplake_datastore.similarity_search(
        "foo", k=1, distance_metric=distance_metric
    )
    assert output == [Document(page_content="foo", metadata={"page": "0"})]

    tql_query = f"SELECT * WHERE " f"id=='{deeplake_datastore.dataset['ids'][0]}'"

    output = deeplake_datastore.similarity_search(
        query="foo", tql_query=tql_query, k=1, distance_metric=distance_metric
    )
    assert len(output) == 1


def test_similarity_search_by_vector(
    deeplake_datastore: DeeplakeVectorStore, distance_metric: str
) -> None:
    """Test similarity search by vector."""
    embeddings = DeterministicFakeEmbedding(size=384).embed_documents(
        ["foo", "bar", "baz"]
    )
    output = deeplake_datastore.similarity_search_by_vector(
        embeddings[1], k=1, distance_metric=distance_metric
    )
    assert output == [Document(page_content="bar", metadata={"page": "1"})]
    deeplake_datastore.delete_dataset()


def test_similarity_search_with_score(
    deeplake_datastore: DeeplakeVectorStore, distance_metric: str
) -> None:
    """Test similarity search with score."""
    deeplake_datastore.dataset.summary()
    output, score = deeplake_datastore.similarity_search_with_score(
        "foo", k=1, distance_metric=distance_metric
    )[0]
    assert output == Document(page_content="foo", metadata={"page": "0"})
    if distance_metric == "cos":
        assert math.isclose(score, 1.0, rel_tol=1e-5)
    else:
        assert math.isclose(score, 0.0, rel_tol=1e-5)
    deeplake_datastore.delete_dataset()


def test_similarity_search_with_filter(
    deeplake_datastore: DeeplakeVectorStore, distance_metric: str
) -> None:
    """Test similarity search."""

    output = deeplake_datastore.similarity_search(
        "foo",
        k=1,
        distance_metric=distance_metric,
        filter={"metadata": {"page": "1"}},
    )
    assert output == [Document(page_content="bar", metadata={"page": "1"})]
    deeplake_datastore.delete_dataset()


def test_max_marginal_relevance_search(deeplake_datastore: DeeplakeVectorStore) -> None:
    """Test max marginal relevance search by vector."""

    output = deeplake_datastore.max_marginal_relevance_search("foo", k=1, fetch_k=2)

    assert output == [Document(page_content="foo", metadata={"page": "0"})]

    embeddings = DeterministicFakeEmbedding(size=384).embed_documents(
        ["foo", "bar", "baz"]
    )
    output = deeplake_datastore.max_marginal_relevance_search_by_vector(
        embeddings[0], k=1
    )

    assert output == [Document(page_content="foo", metadata={"page": "0"})]
    deeplake_datastore.delete_dataset()


def test_delete_dataset_by_ids(deeplake_datastore: DeeplakeVectorStore) -> None:
    """Test delete dataset."""
    id = deeplake_datastore.dataset['ids'][0]
    deeplake_datastore.delete(ids=[id])
    assert (
        deeplake_datastore.similarity_search(
            "foo", k=1, filter={"metadata": {"page": "0"}}
        )
        == []
    )
    assert len(deeplake_datastore.dataset) == 2

    deeplake_datastore.delete_dataset()


def test_vectorstore_creation():
    vectorstore = DeeplakeVectorStore("mem://test_vectorstore_creation")
    assert vectorstore is not None
    assert vectorstore.__class__.__name__ == "DeeplakeVectorStore"


def test_data_insertion():
    vectorstore = DeeplakeVectorStore(
        "mem://test_data_insertion",
        embedding_function=DeterministicFakeEmbedding(size=384),
    )
    ids = vectorstore.add_texts(
        ["hello world", "goodbye world"], metadatas=[{"language": "en"}] * 2
    )
    assert len(ids) == 2
    docs = vectorstore.get_by_ids(ids)
    assert docs[0].page_content == "hello world"
    assert docs[1].page_content == "goodbye world"
    assert docs[0].metadata["language"] == "en"
    assert docs[1].metadata["language"] == "en"


def test_search():
    def download_and_chunk_text():
        url = "https://www.gutenberg.org/files/4280/4280-0.txt"
        response = requests.get(url)
        text = response.text.replace("\r\n", "\n")

        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip() != ""]
        return paragraphs

    texts = download_and_chunk_text()
    vectorstore = DeeplakeVectorStore.from_texts(
        dataset_path="mem://test_search",
        texts=texts,
        embedding=DeterministicFakeEmbedding(size=384),
    )
    assert len(vectorstore) == len(texts)
    results = vectorstore.similarity_search("how we think", k=5)
    assert len(results) == 5

    vectorstore = DeeplakeVectorStore(
        "mem://test_search_2", embedding_function=DeterministicFakeEmbedding(size=384)
    )
    ids = vectorstore.add_texts(texts)
    assert len(ids) == len(texts)
    results = vectorstore.similarity_search("how we think", k=5)
    assert len(results) == 5
    vectorstore.delete(ids)
    assert len(vectorstore) == 0
    results = vectorstore.similarity_search("how we think", k=5)
    assert len(results) == 0

    docs = [Document(page_content=content) for content in texts]
    vectorstore.add_documents(docs)
    assert len(vectorstore) == len(texts)
    vectorstore.dataset.summary()
    results = vectorstore.similarity_search("how we think", k=5)
    assert len(results) >= 3 # The number must be greater than 3 because the text is in the dataset
