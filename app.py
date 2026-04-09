import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load API key
load_dotenv()

# Load document
loader = TextLoader("data/sample.txt")
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings()

# Store in vector DB
vectorstore = Chroma.from_documents(docs, embeddings)

# Create retriever
retriever = vectorstore.as_retriever()

# LLM
llm = ChatOpenAI(temperature=0)

# Ask question
query = input("Ask a question: ")

# 🔍 Retrieve relevant docs
retrieved_docs = retriever.invoke(query)

# 🧱 Build context
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

# 🤖 Generate answer
prompt = f"""
Answer the question using ONLY the context below.
If you don't know, say you don't know.

Context:
{context}

Question:
{query}
"""
response = llm.invoke(prompt)
answer = response.content

# 🛑 Hallucination check
check_prompt = f"""
Check if the answer is supported by the context.

Context:
{context}

Answer:
{answer}

Respond ONLY with:
YES or NO
"""
check = llm.invoke(check_prompt).content

# 🖨️ Output
print("\nAnswer:", answer)
print("Grounded:", check)

# (Optional) show sources
print("\n--- Retrieved Context ---")
for doc in retrieved_docs:
    print("-", doc.page_content)