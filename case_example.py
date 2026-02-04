# case_examples.py

from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

from retriever_setup import case_prep_retriever
from llm_file import llm

# ------------------ Prompt Template ------------------

example_prompt = ChatPromptTemplate.from_messages([
    ("system",
    """
    You are a consulting trainer. A candidate wants real case examples to understand how business cases are structured.

    Use the retrieved context from consulting casebooks to:
    - Provide relevant case examples.
    - Mention industries involved.
    - Highlight business issues tackled.
    - Give short summaries with key lessons or frameworks.
    """),
    MessagesPlaceholder("chat_history"),
    ("system", "Retrieved context:\n{context}"),
    ("human", "{question}")
])

# ------------------ LCEL Chain Builder ------------------

def load_case_examples_chain():
    from operator import itemgetter  # Ensure itemgetter is available in function scope

    # 1. Define how documents are combined
    def combine_docs(docs):
        """Join retrieved docs into a single context block."""
        return "\n\n".join([d.page_content for d in docs])

    # 2. LCEL retrieval pipeline - Fixed to use RunnablePassthrough.assign to preserve inputs
    retrieval_pipeline = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | case_prep_retriever | combine_docs
        )
        | example_prompt
        | llm
        | StrOutputParser()
    )

    # 3. Persistent Memory Store
    store = {}

    def get_memory(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain_with_memory = RunnableWithMessageHistory(
        retrieval_pipeline,
        get_memory,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

    return chain_with_memory
