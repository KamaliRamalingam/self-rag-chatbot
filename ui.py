import streamlit as st
import os
import time
import uuid

from app import ask_question

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Self-RAG Chatbot", layout="wide")

# ===============================
# SESSION STATE
# ===============================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "recent" not in st.session_state:
    st.session_state.recent = []

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# ===============================
# HELPERS
# ===============================
def stream_text(text):
    placeholder = st.empty()
    output = ""
    for word in text.split():
        output += word + " "
        placeholder.markdown(output)
        time.sleep(0.01)
    return output


def format_answer(answer, status):
    if status == "refusal":
        return "The answer is not in the provided documents."

    answer = answer.strip()

    # Add line breaks before numbered points
    for i in range(1, 15):
        answer = answer.replace(f"{i}. ", f"\n{i}. ")

    # Clean spacing
    lines = [line.strip() for line in answer.split("\n") if line.strip()]

    return "\n".join(lines)


def render_status(status):
    if status == "grounded":
        st.markdown("🟢 **Answer supported by documents**")
    elif status == "unsupported":
        st.markdown("🟡 **Partially supported / uncertain**")
    elif status == "refusal":
        st.markdown("⚪ **No relevant information found in documents**")


def render_confidence(confidence):
    pct = int(confidence * 100)

    if confidence > 0.8:
        color = "#16a34a"
    elif confidence > 0.5:
        color = "#f59e0b"
    else:
        color = "#ef4444"

    st.markdown(f"""
    <span style="background:{color}20;color:{color};padding:4px 10px;border-radius:999px;font-size:12px;">
    Confidence {pct}%
    </span>
    """, unsafe_allow_html=True)


def render_sources(docs):
    if not docs:
        return

    with st.expander(f"📚 Sources ({len(docs)})"):
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", 0)

            if os.path.exists(source):
                with open(source, "rb") as f:
                    st.download_button(
                        label=f"📎 {os.path.basename(source)} — Page {page}",
                        data=f,
                        file_name=os.path.basename(source),
                        mime="application/pdf",
                        key=str(uuid.uuid4())
                    )
            else:
                st.write(f"📄 {source} (Page {page})")


# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.title("📄 Self-RAG Chatbot")

    st.markdown("### 📂 Data Source")
    st.caption("Using preloaded documents from /data folder")

    st.markdown("---")

    st.markdown("### 🧪 Try")

    examples = [
        "What is the purpose of the DOT 2022 report?",
        "What are the main strategic goals mentioned in the reports?",
        "Who is Elon Musk?"
    ]

    for i, q in enumerate(examples):
        if st.button(q, key=f"example_{i}"):
            st.session_state.input_text = q

    st.markdown("### 🕘 Recent Questions")

    for i, q in enumerate(reversed(st.session_state.recent)):
        if st.button(q, key=f"recent_{i}"):
            st.session_state.input_text = q

    st.markdown("---")

    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.session_state.recent = []
        st.session_state.input_text = ""
        st.rerun()


# ===============================
# HEADER
# ===============================
st.title("💬 Self-RAG Chatbot")
st.caption("Ask questions from your preloaded document dataset")

if not st.session_state.chat_history:
    st.info("👋 Ask a question to get started.")


# ===============================
# DISPLAY CHAT
# ===============================
for chat in st.session_state.chat_history:

    # USER
    st.markdown(f"""
    <div style="display:flex;justify-content:flex-end;margin-bottom:10px;">
        <div style="background:#d1e7dd;padding:10px 14px;border-radius:12px;max-width:60%;">
        <b>You</b><br>{chat['question']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ASSISTANT
    st.markdown(f"""
    <div style="background:#ffffff;padding:16px;border-radius:12px;border:1px solid #e5e7eb;margin-bottom:6px;line-height:1.7;font-size:15px;">
    {chat["answer"]}
    </div>
    """, unsafe_allow_html=True)

    render_status(chat["status"])
    render_confidence(chat["confidence"])
    render_sources(chat["docs"])

    st.markdown("<br>", unsafe_allow_html=True)


# ===============================
# INPUT
# ===============================
prompt = st.chat_input("Ask anything about your documents...")

if st.session_state.input_text:
    prompt = st.session_state.input_text
    st.session_state.input_text = ""


# ===============================
# HANDLE QUERY
# ===============================
if prompt:

    if prompt not in st.session_state.recent:
        st.session_state.recent.append(prompt)
        st.session_state.recent = st.session_state.recent[-5:]

    st.session_state.chat_history.append({
        "question": prompt,
        "answer": "⏳ Thinking...",
        "docs": [],
        "status": "unsupported",
        "confidence": 0.0
    })

    st.rerun()


# ===============================
# GENERATE ANSWER
# ===============================
if st.session_state.chat_history:

    last = st.session_state.chat_history[-1]

    if last["answer"] == "⏳ Thinking...":

        with st.spinner("🤖 Generating answer..."):
            result = ask_question(last["question"])

        answer = format_answer(result["answer"], result["status"])
        docs = result.get("sources", [])
        status = result.get("status", "unsupported")
        confidence = result.get("confidence", 0.0)

        streamed = stream_text(answer)

        st.session_state.chat_history[-1] = {
            "question": last["question"],
            "answer": streamed,
            "docs": docs,
            "status": status,
            "confidence": confidence
        }

        st.rerun()