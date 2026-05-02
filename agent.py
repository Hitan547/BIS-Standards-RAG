"""
agent.py
LangGraph corrective RAG agent — upgraded for BIS standard recommendation.
Output: ranked list of IS standards with rationale.
Self-corrects up to MAX_RETRIES if hallucination detected.
"""

from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from config import GROQ_API_KEY, GROQ_MODEL, MAX_RETRIES

llm = ChatGroq(
    model=GROQ_MODEL,
    temperature=0,
    api_key=GROQ_API_KEY,
)

# ── System instructions ──
BIS_SYSTEM_PROMPT = """You are a BIS (Bureau of Indian Standards) compliance expert assistant.
Your job is to recommend relevant Indian Standards (IS codes) for building material products.

STRICT RULES:
1. ONLY recommend IS standards that are explicitly mentioned in the CONTEXT below.
2. NEVER invent or guess IS codes. If a code isn't in the context, do NOT mention it.
3. Format your response as a ranked numbered list.
4. For each standard, provide: IS code, title, and 1-2 sentence rationale.
5. Focus on why this standard applies to the specific product described.
"""

BIS_OUTPUT_FORMAT = """
Format each recommendation EXACTLY like this:

1. IS XXXX:YYYY — Title of Standard
   Rationale: Why this standard applies to the described product.

2. IS XXXX:YYYY — Title of Standard
   Rationale: Why this standard applies to the described product.
(continue for all relevant standards found in context)
"""


class RAGState(TypedDict):
    question:          str
    context_chunks:    list
    answer:            str
    validation_result: str
    fail_reason:       str
    retry_count:       int
    chat_history:      list


def _build_context_text(context_chunks: list) -> str:
    """Build context string from retrieved chunks."""
    parts = []
    for i, r in enumerate(context_chunks, 1):
        std_num  = r.get("standard_number", r.get("source", "Unknown"))
        title    = r.get("title", "")
        category = r.get("category", "")
        chunk    = r.get("chunk", "")
        conf     = r.get("confidence", 0)

        header = f"[Standard {i}: {std_num}"
        if title:    header += f" — {title}"
        if category: header += f" | Category: {category}"
        header += f" | Relevance: {conf:.0%}]"

        parts.append(f"{header}\n{chunk}")

    return "\n\n" + ("─" * 60) + "\n\n".join(parts)


def generate_node(state: RAGState) -> dict:
    context_text = _build_context_text(state["context_chunks"])

    # Build chat history
    history_lines = []
    for msg in state.get("chat_history", [])[-6:]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {msg.content}")
    history_text = "\n".join(history_lines) if history_lines else "None"

    # Add correction note on retry
    correction = ""
    if state.get("retry_count", 0) > 0:
        correction = (
            f"\n\n⚠️ CORRECTION REQUIRED: Your previous response was rejected.\n"
            f"Reason: {state.get('fail_reason', 'unverifiable claims')}.\n"
            f"Only cite IS standards that appear in the CONTEXT. Do not invent codes."
        )

    prompt = (
        f"{BIS_SYSTEM_PROMPT}"
        f"{correction}"
        f"\n\nPREVIOUS CONVERSATION:\n{history_text}"
        f"\n\nCONTEXT (Available IS Standards):\n{context_text}"
        f"\n\nPRODUCT DESCRIPTION: {state['question']}"
        f"\n\nRecommend the most relevant BIS standards for this product."
        f"{BIS_OUTPUT_FORMAT}"
        f"\nAnswer:"
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return {"answer": response.content}


def validate_node(state: RAGState) -> dict:
    """
    Validate that:
    1. Every IS code cited in the answer exists in the context.
    2. No imaginary standards are referenced.
    3. The answer addresses the product description.
    """
    import re
    context_text = _build_context_text(state["context_chunks"])

    # Extract IS codes mentioned in the answer
    cited_codes = re.findall(r"IS\s*\d{3,5}(?::\d{4})?", state["answer"], re.IGNORECASE)

    # Extract IS codes available in context
    available_codes = re.findall(r"IS\s*\d{3,5}(?::\d{4})?", context_text, re.IGNORECASE)
    available_set   = {c.replace(" ", "").upper() for c in available_codes}

    # Check for hallucinated codes
    hallucinated = []
    for code in cited_codes:
        normalized = code.replace(" ", "").upper()
        if normalized not in available_set:
            hallucinated.append(code)

    if hallucinated:
        return {
            "validation_result": "FAIL",
            "fail_reason": f"Hallucinated IS codes not in context: {', '.join(hallucinated)}. Only cite codes from the provided context."
        }

    # Also do LLM-based check for other hallucinations
    prompt = (
        "You are a strict hallucination checker for a BIS standards recommendation system.\n\n"
        "Check the ANSWER against the CONTEXT:\n"
        "1. Are all IS codes mentioned in the ANSWER present in the CONTEXT?\n"
        "2. Are the standard titles/descriptions accurate to the CONTEXT?\n"
        "3. Is the rationale grounded in the CONTEXT?\n\n"
        f"CONTEXT:\n{context_text[:3000]}\n\n"
        f"PRODUCT QUERY: {state['question']}\n"
        f"ANSWER: {state['answer']}\n\n"
        "Respond EXACTLY in this format:\n"
        "VERDICT: PASS\nREASON: <one sentence>\n\nor\n\n"
        "VERDICT: FAIL\nREASON: <one sentence explaining what is hallucinated>"
    )

    result = llm.invoke([HumanMessage(content=prompt)])
    text   = result.content.strip()

    verdict = "PASS" if "VERDICT: PASS" in text.upper() else "FAIL"
    reason  = ""
    for line in text.splitlines():
        if line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()
            break

    return {"validation_result": verdict, "fail_reason": reason}


def increment_retry_node(state: RAGState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def route_after_validation(state: RAGState) -> str:
    if (
        state["validation_result"] == "FAIL"
        and state.get("retry_count", 0) < MAX_RETRIES
    ):
        return "retry"
    return "done"


def _build_graph():
    g = StateGraph(RAGState)
    g.add_node("generate",        generate_node)
    g.add_node("validate",        validate_node)
    g.add_node("increment_retry", increment_retry_node)
    g.set_entry_point("generate")
    g.add_edge("generate", "validate")
    g.add_conditional_edges(
        "validate",
        route_after_validation,
        {"retry": "increment_retry", "done": END},
    )
    g.add_edge("increment_retry", "generate")
    return g.compile()


_rag_graph = _build_graph()


def run_rag_agent(
    question:       str,
    context_chunks: list,
    chat_history:   list = [],
) -> tuple[str, int, str]:
    """
    Run the corrective RAG agent.

    Returns:
        (answer, retry_count, validation_result)
    """
    init_state: RAGState = {
        "question":          question,
        "context_chunks":    context_chunks,
        "answer":            "",
        "validation_result": "",
        "fail_reason":       "",
        "retry_count":       0,
        "chat_history":      chat_history,
    }
    final = _rag_graph.invoke(init_state)
    return final["answer"], final["retry_count"], final["validation_result"]