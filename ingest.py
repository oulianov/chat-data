"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

BASE_PATH = "/home/haxxor/projects/supercache/chat-data"


def ingest_docs():
    """Get documents from web pages."""
    loader = ReadTheDocsLoader(
        f"{BASE_PATH}/karpathy.github.io/2022",
        custom_html_tag=("div", {"class": "page-content"}),
    )
    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open(f"{BASE_PATH}/vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    with open(f"{BASE_PATH}/snippets.txt", "w") as f:
        f.write("\n----\n".join([d.page_content for d in documents]))


if __name__ == "__main__":
    ingest_docs()
