from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from ingestion.load_docs import load_documents

def create_vector_store():
    documents = load_documents()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )

    return vectorstore





