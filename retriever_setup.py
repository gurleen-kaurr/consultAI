from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Loading the same embedding model(for coversion otherwise it wont work)
embedding_model = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2'
)

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)
prepare_retriever = vectorstore.as_retriever(
    search_kwargs={"k":3,"filter": {"part": "prepare_yourself"}}
)

learning_retriever = vectorstore.as_retriever(
    search_kwargs={"k":3,"filter": {"part": "learning"}}
)

case_prep_retriever = vectorstore.as_retriever(
    search_kwargs={"k":3,"filter": {"part": "case_prep"}}
)

print(" Retrievers loaded and filtered successfully.")

# Export retrievers
__all__ = ["prepare_retriever", "learning_retriever", "case_prep_retriever"]
