import os

import gradio as gr
from langchain.chains import RetrievalQA
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.llms import Ollama

VECTOR_DIR = "faiss_index"
embeddings = OllamaEmbeddings(model="llama3")
qa_chain = None


def upload_docs(files):
    global qa_chain
    qa_chain = None  # Reset chain

    try:
        docs = []
        for file in files:
            ext = os.path.splitext(file.name)[1].lower()
            if ext == ".pdf":
                loader = PyPDFLoader(file.name)
            elif ext == ".txt":
                loader = TextLoader(file.name)
            else:
                return f"❌ Unsupported file type: {ext}"

            docs.extend(loader.load())

        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
        vectorstore.save_local(VECTOR_DIR)

        return "✅ Document uploaded and indexed!"
    except Exception as e:
        return f"❌ Upload failed: {str(e)}"


def ask_question(query):
    global qa_chain
    try:
        if not qa_chain:
            vectorstore = FAISS.load_local(
                VECTOR_DIR,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vectorstore.as_retriever()
            llm = Ollama(model="llama3")
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        return qa_chain.run(query)
    except Exception as e:
        return f"❌ Error: {str(e)}"


with gr.Blocks() as app:
    gr.Markdown("## 🧠 Document Q&A — Powered by Ollama, FAISS & LangChain")
    gr.Markdown("Upload your PDF/TXT files, and ask anything.")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(file_types=[".pdf", ".txt"], file_count="multiple", label="📤 Upload your document")
            upload_button = gr.Button("📁 Upload & Index")
            upload_ack = gr.Markdown("")  # Upload status

        with gr.Column():
            summarize_button = gr.Button("🧠 Summarize Document")

            question_input = gr.Textbox(placeholder="💬 Ask a question...", lines=1, show_label=False)

            sample_questions = gr.Dropdown(
                choices=[
                    "What is the main topic of this document?",
                    "List key points from this file.",
                    "Summarize this document in 5 lines.",
                    "Explain concisely what this document is about.",
                ],
                label="📚 Sample Questions",
                interactive=True,
            )

    answer_output = gr.Markdown()

    # Bind events
    upload_button.click(
        fn=lambda: "⏳ Indexing document... Please wait.",
        inputs=None,
        outputs=upload_ack,
    ).then(
        fn=upload_docs,
        inputs=[file_input],
        outputs=[upload_ack]
    )

    summarize_button.click(fn=lambda: ask_question("Summarize the document briefly"), outputs=[answer_output])

    question_input.submit(fn=ask_question, inputs=[question_input], outputs=[answer_output])
    sample_questions.change(fn=ask_question, inputs=[sample_questions], outputs=[answer_output])
    gr.Markdown("---")
    gr.Markdown("💡 Tip: You can re-upload files anytime. Context resets with each new upload.")

app.launch()
