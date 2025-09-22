from typing import Annotated, Literal, List, TypedDict
from core.config import settings
from processor import get_processor

from pydantic import BaseModel
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# --- State ---
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class AgentState(State):
    route: str
    rag: str | None
    sources: List[str] | None


class RouteDecision(BaseModel):
    """Decision model for routing queries"""
    route: Literal["rag", "end"]
    reply: str | None = None


# --- LLMs ---
router_llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model="llama-3.1-8b-instant"
).with_structured_output(RouteDecision)

answer_llm = ChatGroq(
    api_key=settings.GROQ_API_KEY,
    model="llama-3.1-8b-instant"
)


# --- Tools ---
@tool
def rag_search_tool(query: str) -> str:
    """Search KB by semantic similarity or direct page lookup"""
    print(f"\n🔎 Searching for: '{query}'")

    import re

    # Page range
    range_match = re.search(r"pages?\s+(\d+)\s*(?:to|-|–)\s*(\d+)", query, re.IGNORECASE)
    if range_match:
        start, end = map(int, range_match.groups())
        results = []
        for page_number in range(start, end + 1):
            results.extend(get_processor().search_by_page(page=page_number, limit=5))
        if not results:
            return f"No content found for pages {start}–{end}."
        return "\n".join([f"--- Page {r['page_number']} ---\n{r['content']}" for r in results])

    # Single page
    single_page_match = re.search(r"(?:on\s+)?pages?\s+(\d+)", query, re.IGNORECASE)
    if single_page_match:
        page_number = int(single_page_match.group(1))
        results = get_processor().search_by_page(page=page_number, limit=10)
        if not results:
            return f"No content found for page {page_number}."
        return "\n".join([f"--- Page {r['page_number']} ---\n{r['content']}" for r in results])

    # Semantic search
    results = get_processor().search_hybrid(query=query, limit=15)
    if not results:
        return "No relevant information found in the knowledge base."

    formatted_results = []
    for i, r in enumerate(results[:3]):
        chunk_info = f"Source: {r['source_file']} | Page: {r.get('page_number', 'N/A')}"
        formatted_results.append(f"--- Result {i+1} ---\n{chunk_info}\n{r['content']}\n")
    return "\n".join(formatted_results)


# --- Nodes ---
def router_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content.lower()
    system_prompt = """You are a routing assistant.
Decide if the user query should go to RAG search or end the conversation.
Routes: rag | end"""

    decision = router_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Query: {query}"),
    ])

    route = decision.route if decision.route in ["rag", "end"] else "rag"
    print(f"📌 Router decision: {route} for query '{query}'")

    new_state = {**state, "route": route}
    if route == "end" and decision.reply:
        new_state["messages"] = state["messages"] + [AIMessage(content=decision.reply)]
    return new_state


def rag_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    chunks = rag_search_tool(query)
    sources = []

    if chunks:
        import re
        for chunk in chunks.split("\n\n"):
            match = re.search(r"Source: ([^\|]+) \| Page: (\d+)", chunk)
            if match:
                sources.append(f"{match.group(1).strip()} (Page {match.group(2)})")
        sources = list(set(sources))

    return {**state, "rag": chunks, "route": "answer", "sources": sources}


def answer_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    context = state.get("rag") or "No relevant context was retrieved from the knowledge base."
    sources = state.get("sources", [])

    system_prompt = """You are a helpful assistant called Merve.
Answer questions using the provided context from the knowledge base."""

    reply = answer_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
    ])

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=reply.content)],
        "sources": sources,
        "route": "end",
    }


def from_router(state: AgentState) -> str:
    return state["route"]


# --- Workflow ---
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("rag_lookup", rag_node)
workflow.add_node("answer", answer_node)
workflow.set_entry_point("router")
workflow.add_conditional_edges("router", from_router, {
    "rag": "rag_lookup",
    "end": END,
})
workflow.add_edge("rag_lookup", "answer")
workflow.add_edge("answer", END)

agent = workflow.compile(checkpointer=MemorySaver())


def agent_responder(query: str, thread_id: str = "thread-id"):
    try:
        config = {"configurable": {"thread_id": thread_id}}
        result = agent.invoke({"messages": [HumanMessage(content=query)]}, config)

        formatted_messages = [
            {"content": m.content, "type": m.__class__.__name__}
            for m in result.get("messages", [])
        ]
        latest_message = result["messages"][-1] if result.get("messages") else None
        message_content = latest_message.content if latest_message else ""

        return {
            "status": "success",
            "message": message_content,
            "sources": result.get("sources", []),
            "query": query,
            "route": result.get("route", "unknown"),
            "messages": formatted_messages,
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error: {str(e)}",
            "sources": [],
            "query": query,
            "route": "error",
            "messages": [],
        }
