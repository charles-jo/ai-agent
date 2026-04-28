from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from .config import settings
from .models import Message

_llm = ChatOpenAI(
    model=settings.llm_model,
    base_url=settings.vllm_base_url,
    api_key=settings.vllm_api_key,
    max_tokens=settings.llm_max_tokens,
    temperature=settings.llm_temperature,
)

rewrite_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are a search query optimizer. Given a conversation history and a follow-up question, "
         "rewrite it as a standalone search query for a vector database. "
         "Output only the rewritten query, nothing else."),
        MessagesPlaceholder("history"),
        ("human", "{query}"),
    ])
    | _llm
    | StrOutputParser()
)

rag_chain = (
    ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert IaC assistant. Answer using only the provided context. "
         "Be concise and accurate. If the context lacks information, say so.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{query}"),
    ])
    | _llm
    | StrOutputParser()
)


def to_lc_messages(history: list[Message]) -> list:
    return [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in history
    ]
