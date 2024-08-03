import os
from dotenv import load_dotenv
import gradio as gr
from uuid import uuid4
from src.feature.chat.chat_handler import handle_chat
from src.feature.qa.qa_handler import handle_qa

load_dotenv()

# Set feature name (chat|qa)
feature_name = os.environ.get("FEATURE_NAME")

def answer_question(question):
    if feature_name == "chat":
        print("chat")
        session_id = str(uuid4())
        return handle_chat(session_id, question)
    else:
        return handle_qa(question)

# Set title based on feature name
title = f"Movie History {'Chat' if feature_name == 'chat' else 'Q&A'} (gen-ai)"

iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask a question about movie history..."),
    outputs="text",
    title=title,
    description=(
        "Ask any question about film history and get an answer based on our knowledge base.\n\n"
        "Example questions:\n"
        "- Which brothers screened Workers Leaving the Lumière Factory in 1895?\n"
        "- What year was 'The Godfather' released?\n"
        "- Who starred as the lead character in 'Casablanca'?\n"
        "- How did Inception influence the sci-fi genre?\n"
        "- James Cameron movies list\n"
    )
)

if __name__ == "__main__":
    iface.launch()
