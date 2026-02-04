from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough


def load_prepare_chain(llm, prepare_retriever):

    #LCEL Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system",
        """
        You are a consulting coach. Use ONLY the retrieved context below to help answer the user's question.
        If the context does not contain the answer, say that the information is not available.
        Be concise, structured, and clear.
        """),
        MessagesPlaceholder("chat_history"),
        ("system", "Retrieved context:\n{context}"),
        ("human", "{question}")
    ])

    #combining retrieved docs
    def combine_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # ---- Retrieval + Prompt + LLM Pipeline ----
    # Fixed: Use RunnablePassthrough.assign to add context while preserving inputs
    pipeline = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | prepare_retriever | combine_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    # ---- Persistent Memory Store ----
    store = {}

    def get_memory(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain_with_memory = RunnableWithMessageHistory(
        pipeline,
        get_memory,
        input_messages_key="question",
        history_messages_key="chat_history"
    )

    return chain_with_memory
