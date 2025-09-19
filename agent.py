from typing import Annotated, Literal, List, TypedDict
from core.config import settings
from processor import get_processor

# from documents.pc import processor
from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class AgentState(State):
    route: str
    rag: str | None
    sources: List[str] | None


class RouteDecision(BaseModel):
    """Decision model for routing queries"""

    route: Literal["rag", "answer", "end", "document_info", "greeting"]
    reply: str | None = None

    class Config:
        json_schema_extra = {
            "examples": [
                {"route": "rag", "reply": None},
                {"route": "document_info", "reply": None},
                {"route": "greeting", "reply": None},
                {"route": "end", "reply": "Goodbye! Have a great day!"},
            ]
        }


router_llm = ChatGroq(
    api_key=settings.GROQ_API_KEY, model="llama-3.1-8b-instant"
).with_structured_output(RouteDecision)

answer_llm = ChatGroq(api_key=settings.GROQ_API_KEY, model="llama-3.1-8b-instant")


@tool
def greeting_tool(query: str) -> str:
    """Respond to greetings or small talks in a friendly way"""
    greetings = {
        "hi": "👋 Hello! How can I help you today?",
        "hello": "👋 Hi there! How’s your day going?",
        "hey": "Hey! What’s up?",
        "good morning": "🌅 Good morning! Ready to get started?",
        "good afternoon": "☀️ Good afternoon! How’s everything?",
        "good evening": "🌙 Good evening! Hope your day went well!",
        "how are you": "I’m doing great, thanks for asking! How are you doing?",
    }
    
    q = query.lower.strip()
    for key, response in greetings.items():
        if key in q:
            return response
    return "Hi I'm Merve, your document assistant, How can I help you today?"

@tool
def rag_search_tool(query: str) -> str:
    """Search KB by semantic similarity or direct page lookup"""
    print(f"\nSearching for: '{query}'")

    # Detect page reference
    import re

    range_match = re.search(r"pages?\s+(\d+)\s*(?:to|-|–)\s*(\d+)", query, re.IGNORECASE)
    if range_match:
        start, end = map(int, range_match.groups())
        results = []
        for page_number in range(start, end + 1):
            page_results = get_processor().search_by_page(page=page_number, limit=5)
            results.extend(page_results)
        if not results:
            return f"No content found for pages {start}–{end}."
        formatted = [f"--- Page {r['page_number']} ---\n{r['text']}" for r in results]
        return "\n".join(formatted)
    
    single_page_match = re.search(r"(?:on\s+)?pages?\s+(\d+)", query, re.IGNORECASE)
    if single_page_match:
        page_number = int(single_page_match.group(1))
        results = get_processor().search_by_page(page=page_number, limit=10)
        if not results:
            return f"No content found for page {page_number}."
        formatted = [f"--- Page {r['page_number']} ---\n{r['text']}" for r in results]
        return "\n".join(formatted)

    # Otherwise semantic search
    results = get_processor().search_hybrid(query=query, limit=25)
    if not results:
        return "No relevant information found in the knowledge base."

    formatted_results = []
    for i, r in enumerate(results[:3]):
        chunk_info = f"Source: {r['source_file']} | Page: {r.get('page_number', 'N/A')}"
        formatted_results.append(f"--- Result {i+1} ---\n{chunk_info}\n{r['text']}\n")
    return "\n".join(formatted_results)


@tool
def get_document_info() -> str:
    """Get information about available documents in the knowledge base"""
    try:
        # Peek into a larger batch to see documents
        results = get_processor().collection.peek(limit=500)

        if not results or "ids" not in results or not results["ids"]:
            return "Knowledge base is empty. No documents have been added yet."

        doc_stats = {}
        total_chunks = len(results["ids"])

        for meta in results.get("metadatas", []):
            if not meta:
                continue

            source = meta.get("source_file", "Unknown")
            page = meta.get("page_number")

            if source not in doc_stats:
                doc_stats[source] = {"chunks": 0, "pages": set()}

            doc_stats[source]["chunks"] += 1
            if page and str(page).isdigit():
                doc_stats[source]["pages"].add(int(page))

        # Format output
        lines = ["📑 Documents in knowledge base:\n"]
        for source, stats in doc_stats.items():
            if stats["pages"]:
                min_page, max_page = min(stats["pages"]), max(stats["pages"])
                page_range = f"Pages {min_page}–{max_page} ({len(stats['pages'])} unique)"
            else:
                page_range = "No page info"
            lines.append(f"• {source} — {stats['chunks']} chunks | {page_range}")

        lines.append(f"\nTotal chunks stored: {total_chunks}")
        return "\n".join(lines)

    except Exception as e:
        return f"Error retrieving document information: {str(e)}"



def router_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content.lower()
    ## Add greetings here no need for node
    system_prompt = """You are a routing assistant. Analyze the query and choose the appropriate route:

- Choose "document_info" for questions about what documents are available, what's in storage, listing documents
- Choose "rag" for questions about specific content, facts, or information that might be in documents
- Choose "answer" for general questions, opinions, calculations, or simple greetings  
- Choose "end" for clear goodbyes like "bye", "goodbye", "see you later", "Nice chatting with you" (include a farewell message)

Look for these patterns:
- Document info: "what documents", "what's in storage", "what files", "list documents", "What is in my knowledge base"
- Greetings: for greetings, pleasantries, small talk ("hi", "hello", "hey", "how are you", "good morning")
- RAG search: specific factual questions, "what is", "tell me about", "explain", page references
- General: greetings, opinions, calculations, "how are you"

Format your response as a JSON object with:
- "route": exactly "rag", "answer", "end", "document_info" or "greeting" 
- "reply": farewell message if route is "end", otherwise null

IMPORTANT: You must ONLY output valid JSON that strictly matches the schema. Do not include explanations, markdown, or extra text.

"""


    decision = router_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Query: {query}\nChoose the appropriate route."),
        ]
    )

    valid_routes = ["rag", "answer", "end", "document_info"]
    route = decision.route if decision.route in valid_routes else "answer"

    print(f"Router decision: {route} for query: '{query}'")

    new_state = {**state, "route": route}

    if route == "end" and decision.reply:
        new_state["messages"] = state["messages"] + [AIMessage(content=decision.reply)]

    return new_state


def greeting_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    response = greeting_tool.invoke({
        "query": query
    })
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response)],
        "sources": [],
        "route": "end"
    }

def document_info_node(state: AgentState) -> AgentState:
    doc_info = get_document_info.invoke({})
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=doc_info)],
        "route": "end",
        "sources": [],
    }


def rag_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    chunks = rag_search_tool.invoke({"query": query})
    
    sources = []
    if chunks:
        import re
        for chunk in chunks.split("\n\n"):
            source_match = re.search(r"Source: ([^\|]+) \| Page: (\d+)", chunk)
            if source_match:
                source_file, page = source_match.groups()
                sources.append(f"{source_file.strip()} (Page {page})")
        sources = list(set(sources))

    return {
        **state,
        "rag": chunks,
        "route": "answer",
        "sources": sources
    }


def answer_node(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    context = state.get("rag") or ""
    sources = state.get("sources", [])

    # Enhanced system prompt for better responses
    system_prompt = """You are a helpful assistant called Merve. Use the provided context to answer questions accurately.
If referencing document content, include the page number in your response as the {sources}.
Be concise but comprehensive. Focus on information from the provided context."""

    reply = answer_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Question: {query}\n\nContext:\n{context}"),
        ]
    )

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=reply.content)],
        "sources": sources,  # Preserve sources from RAG node
        "route": "end",
    }


def from_router(state: AgentState) -> str:
    """Decide next hop after router"""
    return state["route"]  # "rag", "answer", "end", or "document_info"


# Build the workflow with simplified routing
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("rag_lookup", rag_node)
workflow.add_node("answer", answer_node)
workflow.add_node("document_info", document_info_node)
workflow.add_node("greeting", greeting_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    from_router,
    {
        "rag": "rag_lookup",
        "answer": "answer",
        "end": END,
        "document_info": "document_info",
        "greeting": "greeting",
    },
)
workflow.add_edge("rag_lookup", "answer")
workflow.add_edge("answer", END)
workflow.add_edge("document_info", END)

agent = workflow.compile(checkpointer=MemorySaver())


def agent_responder(
    query: str,
    thread_id: str = "thread-id"
):
    """
    Wrapper function to invoke agent and return structured JSON response
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        result = agent.invoke({"messages" : [HumanMessage(content=query)]},config)
        
        formatted_messages = []
        for msg in result.get("messages", []):
            formatted_messages.append({
                "content": msg.content,
                "type": msg.__class__.__name__
            })
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
            "message": f"Error processing query: {str(e)}",
            "sources": [],
            "query": query,
            "route": "error",
            "messages": []
        }

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "thread‑12"}}
    while True:
        query = input("You: ").strip()
        if query in {"quit", "exit"}:
            break
        
        # Use the new JSON response function
        response = invoke_agent_with_json_response(query, "console-thread")
        
        if response["status"] == "success":
            print(f"Agent: {response['message']}")
            if response["sources"]:
                print(f"Sources: {', '.join(response['sources'])}")
        else:
            print(f"Error: {response['message']}")