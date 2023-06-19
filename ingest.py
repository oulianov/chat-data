"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle

from deta import Deta
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

BASE_PATH = "/home/haxxor/projects/supercache/chat-data"

# Initialize
deta = Deta()
db = deta.Base("docs")


def ingest_docs():
    """Get documents from web pages."""
    loader = ReadTheDocsLoader(
        f"{BASE_PATH}/karpathy.github.io/2022",
        custom_html_tag=("div", {"class": "page-content"}),
    )

    # TODO :
    # for each article :
    # - extract metadata
    # - create summaries
    # - store vectors in a numpy array
    # for the collection :
    # - create summary of summaries
    # push all of the structured data into a deta db

    raw_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()  # type: ignore
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open(f"{BASE_PATH}/vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    with open(f"{BASE_PATH}/snippets.txt", "w") as f:
        f.write("\n----\n".join([d.page_content for d in documents]))


if __name__ == "__main__":
    ingest_docs()
