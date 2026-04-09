# Self-RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built using LangChain and Streamlit. The system answers questions based on a local document collection while ensuring responses are grounded in retrieved context and avoiding hallucinations.

## Overview

This project implements a document question-answering system that combines vector search with language models. It retrieves relevant content from a document store and generates answers strictly based on that context. If the answer is not supported by the documents, the system explicitly refuses to respond.

## Features

- Document-based question answering using vector search
- Grounded response generation with self-evaluation
- Hallucination prevention through validation checks
- Adaptive retrieval strategy:
  - Broad queries aggregate information across documents
  - Specific queries return focused answers
- Source attribution with document references
- Interactive web interface using Streamlit

## Technology Stack

- Python
- LangChain
- OpenAI (Embeddings and LLM)
- Chroma (Vector Database)
- Streamlit

## Project Structure

```
ragapp/
├── app.py        # RAG pipeline (retrieval, generation, validation)
├── ui.py         # Streamlit interface
├── data/         # Input documents
├── db/           # Vector database (generated locally)
├── .gitignore
```

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/KamaliRamalingam/ragapp.git
cd ragapp
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file and add:

```
OPENAI_API_KEY=your_api_key_here
```

### 5. Run the application
```bash
streamlit run ui.py
```

## Example Queries

- What is the purpose of the DOT 2022 report?
- What are the main strategic goals mentioned in the reports?
- Who is Elon Musk? (expected to return a refusal)

## System Workflow

1. Documents are loaded and split into smaller chunks
2. Each chunk is converted into embeddings and stored in a vector database
3. A user query retrieves relevant document chunks
4. The language model generates an answer based on retrieved context
5. A validation step checks whether the answer is supported by the context
6. The system either:
   - Returns a grounded answer with sources
   - Refuses to answer if insufficient information is available

## Key Design Decisions

- Validation step ensures answers are grounded in retrieved context
- Keyword overlap check improves relevance filtering
- Adaptive retrieval balances precision and recall
- Source deduplication prevents misleading multiple citations

## Future Improvements

- Improved retrieval ranking (reranking models)
- Multi-step reasoning across documents
- Streaming responses
- Deployment to cloud platforms

## Author

Kamali Ramalingam
