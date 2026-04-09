# Retrieval-Augmented Generation (RAG) Chatbot with Hallucination Detection

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system that combines document retrieval with a language model to generate accurate and context-aware answers.

It demonstrates a simple, production-style approach to building reliable AI systems.

A validation layer is included to detect whether responses are grounded in retrieved context, reducing hallucinations and improving reliability.

---

## Features
- Semantic search using vector embeddings (ChromaDB)
- Context-aware answer generation using OpenAI
- Hallucination detection (grounded vs not grounded)
- Command-line interface for interaction
- Extensible to multiple document formats

---

## Architecture

User Query  
↓  
Retriever (Vector Store)  
↓  
Context Retrieval  
↓  
LLM Answer Generation  
↓  
Validation (Hallucination Check)


## Tech Stack
- Python
- LangChain
- OpenAI API
- ChromaDB
- python-dotenv

---

## Project Structure

rag-app/
│
├── app.py              # CLI RAG application
├── data/               # Input documents
│   └── sample.txt
├── .env                # API keys (not committed)
├── .gitignore
└── README.md

---

## Setup

### 1. Clone the repository
git clone <your-repo-url>
cd rag-app

### 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

### 3. Install dependencies
pip install langchain openai chromadb python-dotenv langchain-community langchain-openai

### 4. Add API key

Create a `.env` file in the root directory:

OPENAI_API_KEY=your_api_key_here

---

## Running the Application

python app.py

---

## Example

Input:
What is Lennart?

Output:
Answer: Lennart is a platform that helps companies access their internal knowledge.
Grounded: YES

---

## Hallucination Detection

The system validates whether the generated answer is supported by retrieved context:

- YES → Answer is grounded in the data  
- NO → Answer is not supported (potential hallucination)  

---

## Key Learnings
- Built an end-to-end RAG pipeline  
- Integrated vector search with LLMs  
- Implemented validation to reduce hallucinations  

---

## Security
- API keys are stored in `.env`
- `.env` is excluded via `.gitignore`

---

## Future Improvements
- Add chatbot UI (Streamlit)
- Support PDF and multiple documents
- Add conversation memory

---

## Conclusion
This project demonstrates how to build a reliable and explainable AI system using RAG, ensuring responses are grounded in real data rather than generated assumptions.