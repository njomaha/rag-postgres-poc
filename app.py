import streamlit as st
from rag_pipeline import ask

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SQL Chat Assistant",
    page_icon="🗄️",
    layout="wide"
)

# ── HEADER ────────────────────────────────────────────────────────────────────

st.title("🗄️ Chat with your Database")
st.caption("Ask anything about your retail data in plain English")
st.divider()

# ── SESSION STATE ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# ── SIDEBAR ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("💡 Sample Questions")
    sample_questions = [
        "How many records are in each table?",
        "Who are the top 5 customers by total spending?",
        "What is the best selling product category?",
        "Show me all pending orders",
        "Which products are low on stock?",
        "How much revenue did we make in 2024?",
        "Which state has the most customers?",
        "What are the top 3 most sold products?"
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_question = q

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── CHAT HISTORY ──────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sql" in msg:
            with st.expander("🛠️ Generated SQL"):
                st.code(msg["sql"], language="sql")
            if msg.get("results"):
                with st.expander("📊 Raw Data"):
                    st.dataframe(msg["results"], use_container_width=True)

# ── HANDLE INPUT ──────────────────────────────────────────────────────────────

# Get question from chat input or sidebar button
question = st.chat_input("Ask anything about your data...")

# Handle sidebar sample question clicks
if "pending_question" in st.session_state:
    question = st.session_state.pending_question
    del st.session_state.pending_question

# Process the question
if question:
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Run RAG pipeline
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(question)

        st.markdown(response["answer"])

        with st.expander("🛠️ Generated SQL"):
            st.code(response["sql"], language="sql")

        if response["results"]:
            with st.expander("📊 Raw Data"):
                st.dataframe(response["results"], use_container_width=True)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sql": response["sql"],
        "results": response["results"]
    })