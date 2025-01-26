from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from langchain_deeplake import DeeplakeVectorStore
import pytest
import re
import requests


def test_vectorstore_creation():
    vectorstore = DeeplakeVectorStore("mem://test_vectorstore_creation")
    assert vectorstore is not None
    assert vectorstore.__class__.__name__ == "DeeplakeVectorStore"


def test_data_insertion():
    vectorstore = DeeplakeVectorStore(
        "mem://test_data_insertion", embedding_function=FakeEmbeddings(size=384)
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
        embedding=FakeEmbeddings(size=384),
    )
    assert len(vectorstore) == len(texts)
    results = vectorstore.similarity_search("how we think", k=5)
    assert len(results) == 5

    vectorstore = DeeplakeVectorStore("mem://test_search_2", embedding_function=FakeEmbeddings(size=384))
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
    results = vectorstore.similarity_search("how we think", k=5)
    assert len(results) == 5
