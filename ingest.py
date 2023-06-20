"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import markdownify

from deta import Deta
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores.faiss import FAISS


BASE_PATH = "/home/haxxor/projects/supercache/chat-data"

# Initialize
deta = Deta()
db = deta.Base("docs")


class CustomReadTheDocsLoader(ReadTheDocsLoader):
    """Loader that loads ReadTheDocs documentation directory dump."""

    def _clean_data(self, data: str) -> str:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(data, "lxml", **self.bs_kwargs)

        # remove style tags
        style_tags = soup.find_all("style")
        for s in style_tags:
            s.decompose()

        # images
        img_tags = soup.find_all("img")
        for i in img_tags:
            i.decompose()

        # unwrap links
        a_tags = soup.find_all("a")
        for a in a_tags:
            a.unwrap()

        # fetch page-content div (where the content of the article is)
        text = soup.find("div", {"class": "page-content"})

        if text is not None:
            # convert back to HTML string
            text = str(text)
        else:
            text = ""
        # trim empty lines
        return "\n".join([t for t in text.split("\n") if t])

    def _html_to_markdown(self, html_data: str) -> str:
        return markdownify.markdownify(html_data).strip()


def ingest_docs():
    """Get documents from web pages."""
    loader = CustomReadTheDocsLoader(
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
    # Split based on the HTML to make sensible chunks (ex: on new <div> or <p>)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        keep_separator=True,
        separators=[
            # First, try to split along HTML tags
            "</body",
            "</div",
            "</p",
            "</li",
            "<h1",
            "<h2",
            "<h3",
            "<h4",
            "<h5",
            "<h6",
            "<span",
            "</table",
            # "<td",
            # "<th",
            "</ul",
            "<ol",
            "<header",
            "<footer",
            "<nav",
            # Head
            "<head",
            # "<style",
            "<script",
            "<meta",
            "<title",
            "",
        ],
    )
    documents = text_splitter.split_documents(raw_documents)
    # Convert HTML to markdown
    text_documents = []
    for d in documents:
        d.page_content = loader._html_to_markdown(d.page_content)
        # Ignore empty docs
        if d.page_content != "":
            text_documents.append(d)
    documents = text_documents

    embeddings = OpenAIEmbeddings()  # type: ignore
    vectorstore = FAISS.from_documents(text_documents, embeddings)

    # Save vectorstore
    with open(f"{BASE_PATH}/vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

    with open(f"{BASE_PATH}/snippets.txt", "w") as f:
        f.write("\n----\n".join([d.page_content for d in documents]))


if __name__ == "__main__":
    ingest_docs()
