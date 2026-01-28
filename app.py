"""Gradio interface for Document Q&A Chatbot."""

import gradio as gr
from src.rag import process_pdf, chat, qa_system


with gr.Blocks(title="Document Q&A Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # Document Q&A Chatbot

        Upload a PDF document and ask questions about its content using RAG (Retrieval Augmented Generation).

        **How it works:**
        1. Upload a PDF document
        2. The system creates embeddings and stores them in a vector database
        3. Ask questions - the system retrieves relevant chunks and generates answers
        4. Conversation memory maintains context across questions
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath"
            )
            upload_btn = gr.Button("Process PDF", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)

            clear_btn = gr.Button("Clear Conversation", variant="secondary")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="Chat with your document",
                height=400
            )
            msg = gr.Textbox(
                label="Ask a question",
                placeholder="What is this document about?",
                lines=2
            )
            send_btn = gr.Button("Send", variant="primary")

    # Example questions
    gr.Examples(
        examples=[
            ["What are the main topics covered in this document?"],
            ["Can you summarize the key points?"],
            ["What examples are given to illustrate the concepts?"],
        ],
        inputs=msg,
    )

    # Event handlers
    upload_btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=status
    )

    def respond(message, chat_history):
        if not message.strip():
            return "", chat_history
        bot_message = chat(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    send_btn.click(respond, [msg, chatbot], [msg, chatbot])

    def clear_chat():
        qa_system.clear_memory()
        return []

    clear_btn.click(clear_chat, outputs=chatbot)


if __name__ == "__main__":
    demo.launch()
