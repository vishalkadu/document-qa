# 🧠 AI Document Chatbot — LangChain + FAISS + Ollama + Gradio

Unlock insights from your documents using a private, local AI-powered Q&A system. This app allows you to **chat with
your own PDFs or text files** using the power of **LangChain**, **FAISS**, **Ollama (LLaMA3)**, and a beautiful **Gradio
interface** — all running locally!

---

## 🚀 Key Features

✅ **Upload & chat with your documents** (PDF/TXT)  
✅ **Local vector search** using FAISS (privacy-first)  
✅ **Ollama LLaMA3 integration** — no API keys or cloud  
✅ **Built with LangChain for robust chaining**  
✅ **Minimal UI with Gradio** — clean and interactive  
✅ **Fast, accurate answers** or one-click document summarization

---

## 🧰 Tech Stack

- ⚙️ [LangChain](https://github.com/langchain-ai/langchain) – document loading, text splitting, chaining
- 🧠 [FAISS](https://github.com/facebookresearch/faiss) – efficient vector similarity search
- 🦙 [Ollama](https://ollama.com/) – run open LLMs like LLaMA3 on your machine
- 🖼️ [Gradio](https://www.gradio.app/) – simple and powerful web UI

---

## 🛠️ Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/vishalkadu/document-qa.git
cd document-qa-app
```

### 2. Set Up Python Environment

```bash
# python 3.12 or higher is required
python3 -m venv venv
source vk_env/bin/activate      # Windows: vk_env\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Install Ollama 🦙 [Ollama](https://ollama.com/)

```bash
# DOWNLOAD https://ollama.com/download
# DOCS https://github.com/ollama/ollama/tree/main/docs
# Ollama lets you run models like LLaMA3 locally (no API key required)
```

### 5. Download LLaMA3 Model

```bash
# Download the LLaMA3 model using Ollama
ollama pull llama3
```

### 5. Run the App

```bash
# v1.py v2.py , v2 is the latest version
python -m v2
```

### 6. Open the App in Your Browser

Open your browser and go to `http://localhost:7860` to access the app.

## 📝 Usage Instructions

1. **Upload your documents**: Click the "📁 Upload & Index" button to upload your PDF or TXT files.
2. **Ask questions**: Type your questions in the input box and hit "Enter" to get answers from the model.
3. **Summarize**: Click the "🧠 Summarize Document" button to get a summary of the uploaded document.
4. **Sample questions**: Use the provided sample questions to test the model's capabilities.




<!--
SEO TAGS:
AI Document Chatbot, LangChain, FAISS, Ollama, Gradio, LLaMA3, Local AI,
Document Q&A, Vector Search, Privacy-first AI, Open LLMs,
Document Summarization, Interactive UI, Python, Machine Learning,
Natural Language Processing, NLP, Chatbot, Conversational AI,
Text Analysis, Document Analysis, Data Science, Data Engineering
-->
