import os
from dotenv import load_dotenv

from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredHTMLLoader
)
from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===============================
# CONFIG
# ===============================
DATA_PATH = "data"
DB_PATH = "db"

# ===============================
# LOAD ENV
# ===============================
load_dotenv()

# ===============================
# INIT MODELS
# ===============================
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0)

# ===============================
# LOAD OR BUILD VECTOR DB
# ===============================
if os.path.exists(DB_PATH):
    print("⚡ Loading existing vector database...")
    vectorstore = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )
else:
    print("📥 Building vector database (first time only)...")

    documents = []

    for file in os.listdir(DATA_PATH):
        path = os.path.join(DATA_PATH, file)

        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".txt"):
            loader = TextLoader(path)
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif file.endswith(".html"):
            loader = UnstructuredHTMLLoader(path)
        else:
            continue

        docs = loader.load()
        documents.extend(docs)

        print(f"Loaded file: {file}")

    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = text_splitter.split_documents(documents)

    print(f"\nTotal chunks created: {len(docs)}")

    # Create DB
    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=DB_PATH
    )
    vectorstore.persist()

    print("✅ Vector database created.")

# ===============================
# SELF-RAG QUERY
# ===============================
query = input("\nAsk a question: ")

max_attempts = 3
all_answers = []
final_docs = []
is_grounded = False  # 🔥 IMPORTANT FIX

# ===============================
# SELF-RAG LOOP
# ===============================
for attempt in range(max_attempts):
    print(f"\n🔁 Attempt {attempt + 1}")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 10 + attempt * 5}
    )

    retrieved_docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # ===============================
    # GENERATE
    # ===============================
    prompt = f"""
    You are a helpful assistant.

    Answer the question using ONLY the context below.

    If the context contains partial information, provide the best possible answer.

    If the context is not sufficient, respond with:
    "Answer not fully available in the retrieved context."

    Do NOT make up information.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)
    answer = response.content.strip()

    # ===============================
    # CLEAN (DEDUP BY GOAL)
    # ===============================
    cleaned_lines = []
    seen = set()

    for line in answer.split("\n"):
        key = line.split(":")[0].strip()

        if key not in seen:
            cleaned_lines.append(line)
            seen.add(key)

    answer = "\n".join(cleaned_lines)

    # Store answer for aggregation
    all_answers.append(answer)

    # ===============================
    # EVALUATE (RELAXED)
    # ===============================
    check_prompt = f"""
    Check if the answer is reasonably supported by the context.

    If the answer is relevant and mostly based on the context, respond YES.

    If the answer is completely unrelated or fabricated, respond NO.

    Context:
    {context}

    Answer:
    {answer}

    Respond ONLY with YES or NO.
    """

    check = llm.invoke(check_prompt).content.strip()

    print("Answer:", answer)
    print("Grounded:", check)

    # ===============================
    # DECISION
    # ===============================
    if check == "YES" and "not found" not in answer.lower():
        final_docs = retrieved_docs
        is_grounded = True  # 🔥 CRITICAL FIX
        break

    print("⚠️ Not grounded → refining retrieval...")

# ===============================
# FINAL AGGREGATION
# ===============================
combined_answer = "\n".join(all_answers)

cleaned_lines = []
seen = set()

for line in combined_answer.split("\n"):
    key = line.split(":")[0].strip()

    if key not in seen:
        cleaned_lines.append(line)
        seen.add(key)

final_answer = "\n".join(cleaned_lines)

# ===============================
# FINAL OUTPUT (FIXED)
# ===============================
print("\n======================")
print("FINAL ANSWER")
print("======================")

if final_answer.strip() and is_grounded:
    print(final_answer)
    print("\nGrounded: YES")
else:
    print("Answer not fully supported by the retrieved documents.")
    print("\nGrounded: NO")

# ===============================
# SOURCES
# ===============================
print("\n--- Sources ---")
for doc in final_docs:
    source = doc.metadata.get("source", "Unknown")
    page = doc.metadata.get("page", "N/A")

    print(f"[Source: {source}, Page: {page}]")
    print(doc.page_content[:200], "...\n")