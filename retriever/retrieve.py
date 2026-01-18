from embeddings.vector_store import create_vector_store

# Create vector DB once
vectorstore = create_vector_store()

# Convert vector DB into a retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

def retrieve_documents(query: str):
    """
    Given a user query, return the most relevant documents.
    """
    return retriever.invoke(query)

