# case_prep_chain.py

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough

from llm_file import llm  # Your LLM


case_prompts = {
    "Profitability": "A retail chain has seen a 15% drop in net profits...",
    "Market Entry": "A European beverage company wants to enter the Indian market...",
    "M&A": "Your client is considering acquiring a smaller competitor...",
    "Growth Strategy": "A SaaS company wants to increase revenue by 30% in 2 years..."
}


def load_case_prep_chain():

    system_prompt = """
    You are a consulting case interviewer for a top-tier firm (e.g., McKinsey, BCG, Bain).
    Your job is to simulate a real case interview with a candidate.

    During the interview:
    - Start with a well-defined business case prompt.
    - Ask relevant follow-up questions based on the candidate's responses.
    - Keep questions concise and professional.
    - Follow a structured flow: clarifying → framework → analysis → recommendations.
    - Maintain the tone and behavior of an actual interviewer.

    If the candidate asks for clarifications or data, provide concise, realistic answers.
    Do not reveal solutions unless explicitly asked.

    When the user types 'end' or 'done', switch to evaluation mode:
      - Provide feedback under:
          1. Structure
          2. Problem Solving
          3. Communication
          4. Business Acumen
      - Give a 10/10 score for each category
      - Final verdict: Pass / Needs Improvement
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    pipeline = prompt | llm | StrOutputParser()

    store = {}

    def get_memory(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    chain_with_memory = RunnableWithMessageHistory(
        pipeline,
        get_memory,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    return chain_with_memory

__all__ = ["load_case_prep_chain"]