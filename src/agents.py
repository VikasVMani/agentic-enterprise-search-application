from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # <-- THIS loads OPENAI_API_KEY

from src.chroma_db import hybrid_search


# LLM

llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0
)


# Graph State

class AgentState(TypedDict):
    query: str
    partition: str
    retrieved_docs: List[dict]
    answer: str
    history: str



# Routing Agent

def routing_agent(state: AgentState):
    prompt = f"""
You are a routing agent for an enterprise legal document search system.

Available partitions:
- IBM_PurchaseTerms
- IBM_Standard_Terms_and_Conditions
- International_Program_License_Agreement
- International_Agreement_for_Acquisition_of_Software_Maintenance

Conversation history:
{state.get("history", "")}

User query:
{state["query"]}

Return ONLY the best matching partition name.
"""
    response = llm.invoke(prompt).content.strip()
    
    print('query: ',state["query"])
    print('partitions name: ', response)
    return {
        "partition": response
    }



# Retrieval Agent

def retrieval_agent(state: AgentState):
    results = hybrid_search(
        query=state["query"],
        partition=state["partition"],
        top_k=5,
        alpha=0.6
    )
    # print('query: ',state["query"])
    # print('Results: ',results)
    return {
        "retrieved_docs": results
    }



# Response Generator Agent

def response_generator_agent(state: AgentState):
    docs = state["retrieved_docs"]

    if not docs:
        return {
            "answer": "No relevant information found in the enterprise documents."
        }

    evidence_blocks = []
    citation_map = []

    for i, doc in enumerate(docs):
        evidence_blocks.append(
            f"[{i+1}] {doc['text']}"
        )
        citation_map.append(
            f"[{i+1}] {doc['document_name']}, Page {doc['page_no']}"
        )

    evidence_text = "\n\n".join(evidence_blocks)
    citations_text = "\n".join(citation_map)

    prompt = f"""
You are an enterprise legal AI assistant.
Your task is to answer the given question using the provided evidence.
And use past conversation history if the current question is related to it.

Answer the question using the evidence below and the prior onversation history.

Evidence:
{evidence_text}

Prior Conversation History:
{state["history"]}


Question:
{state["query"]}

"""
    answer = llm.invoke(prompt).content

    final_answer = f"""
{answer}

Citations:
{citations_text}
"""

    return {
        "answer": final_answer
    }



# Summarizer Agent

MAX_HISTORY_WORDS = 1200

def summarizer_agent(state: AgentState):
    prompt = f"""
Summarize the conversation history below.
Preserve key legal facts and user intent.

Conversation:
{state["history"]}

Summary:
"""
    summary = llm.invoke(prompt).content
    return {
        "history": summary
    }


def should_summarize(state: AgentState):
    return len(state.get("history", "").split()) > MAX_HISTORY_WORDS



# Build Graph

def build_agent_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router", routing_agent)
    graph.add_node("retriever", retrieval_agent)
    graph.add_node("generator", response_generator_agent)
    graph.add_node("summarizer", summarizer_agent)

    graph.set_entry_point("router")

    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "generator")

    graph.add_conditional_edges(
        "generator",
        should_summarize,
        {
            True: "summarizer",
            False: END
        }
    )

    graph.add_edge("summarizer", END)

    return graph.compile()
