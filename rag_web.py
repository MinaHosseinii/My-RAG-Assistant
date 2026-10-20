
import uuid

import streamlit as st
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from llama_cpp_adapter import LlamaCppAdapter
from rag_pipeline import (
    rag,
    DATABASE_DIR,
    EMBEDDING_MODEL_NAME,
)


# ============================================================
# 1. Page configuration
# ============================================================

st.set_page_config(
    page_title="My RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. ChatGPT-inspired styling
# ============================================================

st.markdown(
    """
    <style>
    /* Overall page */
    .stApp {
        background-color: var(--background-color);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 7rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        font-size: 0.92rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        min-height: 2.6rem;
        text-align: left;
    }

    /* Chat input */
    [data-testid="stChatInput"] {
        max-width: 850px;
        margin-left: auto;
        margin-right: auto;
    }

    [data-testid="stChatInput"] textarea {
        border-radius: 22px;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }

    /* Dividers and secondary text */
    hr {
        border-color: rgba(128, 128, 128, 0.2);
    }

    /* Hide Streamlit's default footer */
    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. Session state and conversation history
# ============================================================

def create_new_chat():
    chat_id = uuid.uuid4().hex

    st.session_state.conversations[chat_id] = {
        "title": "New chat",
        "messages": [],
    }

    st.session_state.active_chat_id = chat_id


if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None

if (
    st.session_state.active_chat_id not in st.session_state.conversations
):
    create_new_chat()


# ============================================================
# 4. Load the RAG components once
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_qdrant_client():
    return QdrantClient(path=str(DATABASE_DIR))


@st.cache_resource
def load_llm():
    return LlamaCppAdapter(port=8080)


# ============================================================
# 5. Sidebar
# ============================================================

with st.sidebar:
    st.markdown("## 📚 My RAG Assistant")
    st.caption("Your local document assistant")

    if st.button(
        "＋  New chat",
        use_container_width=True,
        type="primary",
    ):
        create_new_chat()
        st.rerun()

    st.divider()

    st.markdown("##### Your conversations")

    conversation_ids = list(
        st.session_state.conversations.keys()
    )
    conversation_ids.reverse()

    for chat_id in conversation_ids:
        conversation = st.session_state.conversations[chat_id]

        title = conversation["title"]

        if len(title) > 32:
            title = title[:29] + "..."

        is_active = (
            chat_id == st.session_state.active_chat_id
        )

        button_label = (
            f"●  {title}" if is_active else f"　{title}"
        )

        if st.button(
            button_label,
            key=f"chat_{chat_id}",
            use_container_width=True,
        ):
            st.session_state.active_chat_id = chat_id
            st.rerun()

    st.divider()

    st.caption("SYSTEM")
    st.caption("• Local Qdrant vector database")
    st.caption("• Hybrid search: dense + BM25")
    st.caption("• Qwen through llama.cpp")

    st.caption(
        "Conversation history is kept in this "
        "Streamlit session."
    )


# ============================================================
# 6. Main chat area
# ============================================================

active_chat = st.session_state.conversations[
    st.session_state.active_chat_id
]

st.markdown(
    """
    <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
        <div style="font-size:2rem; margin-bottom:0.4rem;">
            📚
        </div>
        <h2 style="margin:0; font-weight:600;">
            What would you like to know?
        </h2>
        <p style="color:gray; margin-top:0.6rem;">
            Ask questions about your local documents.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 7. Chat input and answer generation
# ============================================================

prompt = st.chat_input(
    "Message My RAG Assistant...",
)

if prompt and prompt.strip():
    question = prompt.strip()

    if not active_chat["messages"]:
        active_chat["title"] = question[:32]

    active_chat["messages"].append(
        {
            "role": "user",
            "content": question,
        }
    )

    try:
        with st.spinner(
            "Searching your documents and generating an answer..."
        ):
            embedding_model = load_embedding_model()
            client = load_qdrant_client()
            llm = load_llm()

            response = rag(
                question=question,
                embedding_model=embedding_model,
                client=client,
                llm=llm,
                k=5,
            )

        sources = []

        for result in response["retrieved_documents"]:
            payload = result.payload or {}

            sources.append(
                {
                    "filename": payload.get(
                        "filename",
                        "Unknown source",
                    ),
                    "score": float(result.score),
                }
            )

        active_chat["messages"].append(
            {
                "role": "assistant",
                "content": response["answer"],
                "sources": sources,
                "context": response["context"],
            }
        )

    except Exception as error:
        active_chat["messages"].append(
            {
                "role": "assistant",
                "content": (
                    "I couldn't complete your request. "
                    "Please check that the local Qdrant "
                    "database and llama.cpp server are available."
                ),
                "error": str(error),
                "sources": [],
                "context": "",
            }
        )

    st.rerun()


# ============================================================
# 8. Render the conversation
# ============================================================

if not active_chat["messages"]:
    st.markdown(
        """
        <div style="
            text-align:center;
            color:gray;
            font-size:0.9rem;
            padding:1.5rem 0;
        ">
            <p>Try asking about renewable energy, forecasting,</p>
            <p>photovoltaic panels, or electricity markets.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

for index, message in enumerate(active_chat["messages"]):
    role = message["role"]

    with st.chat_message(
        role,
        avatar="🧑" if role == "user" else "📚",
    ):
        st.markdown(message["content"])

        if role == "assistant":
            if message.get("error"):
                with st.expander("Technical details"):
                    st.code(message["error"])

            sources = message.get("sources", [])

            if sources:
                with st.expander(
                    f"📄 Retrieved sources ({len(sources)})"
                ):
                    st.caption(
                        "Ranked by hybrid retrieval. "
                        "The scores are ranking scores, "
                        "not probabilities of correctness."
                    )

                    for source in sources:
                        st.markdown(
                            f"- **{source['filename']}** "
                            f"— RRF score: "
                            f"`{source['score']:.4f}`"
                        )

            context = message.get("context", "")

            if context:
                with st.expander("View retrieved context"):
                    st.text(context)
