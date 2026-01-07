import gradio as gr
from src.chroma_db import ingest_documents
from src.doc_processing import prepare_chunks
from src.agents import build_agent_graph


# Preparing chunks
chunks = prepare_chunks('data')
# Ingesting chunks
ingest_documents(chunks)



# Build Agent Graph

graph = build_agent_graph()


# Streaming Chat Function

def chat_stream(user_message, chat_history, agent_memory):
    """
    chat_history: list of (user, assistant)
    agent_memory: summarized conversation memory (string)
    """

    # Call agent graph (non-streaming internally)
    result = graph.invoke({
        "query": user_message,
        "history": agent_memory
    })

    full_answer = result["answer"]

    # Stream response word-by-word
    partial = ""
    for word in full_answer.split():
        partial += word + " "
        updated_history = chat_history + [(user_message, partial)]
        yield updated_history, render_chat(updated_history), agent_memory

    # Finalize history + memory
    chat_history = chat_history + [(user_message, full_answer)]
    agent_memory += f"\nUser: {user_message}\nAssistant: {full_answer}\n"

    yield chat_history, render_chat(chat_history), agent_memory



# Render Chat as Markdown

def render_chat(history):
    if not history:
        return ""
    return "\n\n".join(
        [f"**User:** {u}\n\n**Assistant:** {a}" for u, a in history]
    )



# Gradio UI

with gr.Blocks() as demo:
    gr.Markdown("## 📄 Enterprise Legal Search Assistant (Streaming)")

    chat_state = gr.State([])
    agent_memory = gr.State("")

    chat_display = gr.Markdown()

    msg = gr.Textbox(
        placeholder="Ask a question about IBM contracts or terms…",
        show_label=False
    )

    clear = gr.Button("Clear Chat")

    msg.submit(
        chat_stream,
        inputs=[msg, chat_state, agent_memory],
        outputs=[chat_state, chat_display, agent_memory],
    )

    clear.click(
        lambda: ([], "", ""),
        None,
        outputs=[chat_state, chat_display, agent_memory]
    )


# Launch App

if __name__ == "__main__":
    demo.queue().launch()
