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
# VECTORSTORE LOADER (FIX)
# ===============================
def get_vectorstore():

    if os.path.exists(DB_PATH):
        return Chroma(
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )

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

        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    vectorstore = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=DB_PATH
    )
    vectorstore.persist()

    return vectorstore


# ===============================
# MAIN FUNCTION
# ===============================
def ask_question(query):

    vectorstore = get_vectorstore()  # 🔥 FIX

    max_attempts = 3
    all_answers = []
    final_docs = []
    final_answer = ""
    is_grounded = False

    # 🔥 detect broad queries
    is_broad = "reports" in query.lower()

    for attempt in range(max_attempts):

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 10 + attempt * 5}
        )

        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            continue

        context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # ===============================
        # GENERATE
        # ===============================
        prompt = f"""
        Answer ONLY using the provided context.

        - Do NOT guess.
        - If not found, say nothing useful.

        Context:
        {context}

        Question:
        {query}
        """

        response = llm.invoke(prompt)
        answer = response.content.strip()

        # store for aggregation
        all_answers.append(answer)

        # ===============================
        # VALIDATE
        # ===============================
        check_prompt = f"""
        Is the answer supported by the context?

        Respond ONLY with YES or NO.

        Context:
        {context}

        Answer:
        {answer}
        """

        check = llm.invoke(check_prompt).content.strip()

        # 🔥 keyword match guard
        query_words = [w for w in query.lower().split() if len(w) > 3]
        match_count = sum(word in context.lower() for word in query_words)

        if (
            check == "YES"
            and answer
            and (match_count >= 1 or is_broad)
        ):
            final_docs = retrieved_docs
            is_grounded = True

            # stop early for specific queries
            if not is_broad:
                final_answer = answer
                break

    # ===============================
    # AGGREGATE (ONLY FOR BROAD)
    # ===============================
    if is_broad:
        combined = "\n".join(all_answers)

        seen = set()
        lines = []

        for line in combined.split("\n"):
            clean = line.strip()
            if clean and clean not in seen:
                lines.append(clean)
                seen.add(clean)

        final_answer = "\n".join(lines)

    # ===============================
    # REMOVE DUPLICATE SOURCES
    # ===============================
    unique_sources = {}

    for doc in final_docs:
        src = doc.metadata.get("source", "")
        if src not in unique_sources:
            unique_sources[src] = doc

    clean_docs = list(unique_sources.values())[:4]

    # ===============================
    # FINAL RESPONSE
    # ===============================
    if final_answer and is_grounded:
        return {
            "answer": final_answer,
            "sources": clean_docs,
            "status": "grounded",
            "confidence": 0.85
        }
    else:
        return {
            "answer": "The answer is not in the provided documents.",
            "sources": [],
            "status": "refusal",
            "confidence": 0.0
        }