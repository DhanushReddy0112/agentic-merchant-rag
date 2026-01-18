from typing import TypedDict, List

from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM

from retriever.retrieve import retrieve_documents

# Local LLM
llm = OllamaLLM(model="llama3.1:8b")


# ---------
# STATE
# ---------
class AgentState(TypedDict):
    question: str
    documents: List[str]
    answer: str


# ---------
# NODES
# ---------

def retrieve_node(state: AgentState):
    docs = retrieve_documents(state["question"])
    state["documents"] = [doc.page_content for doc in docs]
    return state


def decide_node(state: AgentState):
    if len(state["documents"]) == 0:
        state["answer"] = "I don't know."
        return state

    prompt = f"""
You are a classifier.

Question:
{state["question"]}

Is this a merchant support related question?
Answer YES or NO only.
"""
    decision = llm.invoke(prompt).strip().upper()

    if decision.startswith("NO"):
        state["answer"] = "I can only help with merchant support related questions."
        return state

    return state


def answer_node(state: AgentState):
    context = "\n".join(state["documents"])

    prompt = f"""
You are a merchant support assistant.

Answer ONLY using the context below.
If the answer is not present, say "I don't know".

Context:
{context}

Question:
{state["question"]}
"""

    state["answer"] = llm.invoke(prompt)
    return state


# ---------
# GRAPH
# ---------

graph = StateGraph(AgentState)

graph.add_node("retrieve", retrieve_node)
graph.add_node("decide", decide_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "decide")

graph.add_conditional_edges(
    "decide",
    lambda state: "answer" if state["answer"] == "" else END,
)

graph.add_edge("answer", END)

merchant_agent = graph.compile()


# ---------
# PUBLIC API
# ---------

def run_agent(question: str) -> str:
    result = merchant_agent.invoke(
        {"question": question, "documents": [], "answer": ""}
    )
    return result["answer"]

