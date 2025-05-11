import gradio as gr
import os
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
import tempfile

VECTOR_DIR = "faiss_index"
embeddings = OllamaEmbeddings(model="llama3")
qa_chain = None

def upload_docs(files):
    docs = []
    qa_chain = None
    for file in files:
        ext = os.path.splitext(file.name)[1].lower()

        # Use file.name directly as it points to the uploaded file path
        if ext == ".pdf":
            loader = PyPDFLoader(file.name)
        elif ext == ".txt":
            loader = TextLoader(file.name)
        else:
            return f"Unsupported file type: {ext}", None

        loaded_docs = loader.load()
        docs.extend(loaded_docs)

    # Split and store in FAISS
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(VECTOR_DIR)

    return "✅ Documents indexed successfully!", True

def ask_question(query):
    global qa_chain
    if not qa_chain:
        vectorstore = FAISS.load_local(
            VECTOR_DIR,
            embeddings=embeddings,
            allow_dangerous_deserialization=True  # 🚨 Only if you're sure the index is safe
        )
        retriever = vectorstore.as_retriever()
        llm = Ollama(model="llama3")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain.run(query)


with gr.Blocks() as app:
    gr.Markdown("## 📄 Document QA with FAISS + LangChain + Ollama (Jupyter Notebook)")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(file_types=[".pdf", ".txt"], file_count="multiple", label="Upload Documents")
            upload_button = gr.Button("Upload & Process")
            upload_status = gr.Textbox(label="Upload Status")

        with gr.Column():
            question_input = gr.Textbox(label="Ask a Question", placeholder="E.g., Summarize the document")
            ask_button = gr.Button("Ask")
            answer_output = gr.Textbox(label="Answer")

    upload_button.click(fn=upload_docs, inputs=[file_input], outputs=[upload_status])
    ask_button.click(fn=ask_question, inputs=[question_input], outputs=[answer_output])

app.launch()
