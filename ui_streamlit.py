import streamlit as st
from datetime import datetime
import os

from ai_news import get_ai_news, chat_ai
from rag.rag_news import index_news_items, rag_answer

st.set_page_config(page_title="Today's AI dosage", layout="wide")

st.markdown(
    """
    <style>
      /* Hide Streamlit chrome */
      [data-testid="stSidebar"] {display: none;}
      [data-testid="stSidebarNav"] {display: none;}
      [data-testid="stToolbar"] {display: none;}
      footer {visibility: hidden;}
      .stDeployButton {display:none;}
      .viewerBadge_container__1QSob {display:none;}

      /* ===== HEADER SPACING ===== */
      .center-title {
        text-align: center;
        margin-top: -15px;   /* move title up a bit */
      }

      /* Reduce top padding of whole page */
      .block-container {
        padding-top: 0.5rem !important;
      }

      /* ===== BACKGROUND ===== */
      [data-testid="stAppViewContainer"] {
        background: #B7C9C9;
      }
      [data-testid="stHeader"] {
        background: transparent !important;
      }
      h1 { color: #2C2C54 !important; }

      /* Radio area */
      [data-testid="stRadio"] > div {
        background: #F1F3F6 !important;
        padding: 8px 12px !important;
        border-radius: 12px !important;
      }

      /* ===== MOVE LEFT IMAGE UP ===== */
      div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(1)
      [data-testid="stImage"] {
        position: relative !important;
        top: -90px !important;  
      }
    </style>
    """,
    unsafe_allow_html=True,
)
# Session defaults
if "chat" not in st.session_state:
    st.session_state.chat = []
if "rag_indexed" not in st.session_state:
    st.session_state.rag_indexed = False
if "last_index_count" not in st.session_state:
    st.session_state.last_index_count = 0

# Title + date
st.markdown("<h1 class='center-title'>Today's AI dosage</h1>", unsafe_allow_html=True)

today = datetime.now().strftime("%b %d %Y")
st.markdown(f'<p class="center-title">{today}</p>', unsafe_allow_html=True)

# Fetch AI news ONCE per rerun (Streamlit reruns the script automatically)
with st.spinner("Fetching AI news..."):
    data = get_ai_news(limit=2, include_summary=True)

articles = data.get("articles", [])[:2]
summary_text = data.get("summary", "")

# ---- 25% LEFT (digest) + 75% RIGHT (chat) ----
left, right = st.columns([1, 3], gap="large")  # 25% / 75%

with left:
    image_path = os.path.join(os.path.dirname(__file__), "assets", "ai_banner.png")
    st.image(image_path, use_container_width=True)

    st.subheader("🤖 Today’s Summary")
    st.write(summary_text if summary_text else "No summary available.")

    st.subheader("📰 Top 2 AI News")
    if not articles:
        st.warning("No articles returned.")
    else:
        for a in articles:
            with st.container(border=True):
                st.markdown(f"**{a.get('title', 'Untitled')}**")
                if a.get("published"):
                    st.caption(a["published"])
                if a.get("url"):
                    st.link_button("Open", a["url"])

with right:
    top_controls = st.container()

    with top_controls:
        mode = st.radio(
            "Chat mode",
            ["Any AI question", "Ask about AI news (RAG)"],
            horizontal=True,
            label_visibility="collapsed"
        )
        question = st.chat_input("Ask anything about AI...")

    st.divider()

    chat_area = st.container()

    # --- CHAT HISTORY ---
    with chat_area:
        for role, msg in st.session_state.chat:
            avatar = "🧑" if role == "user" else "✨"
            with st.chat_message(role, avatar=avatar):
                st.write(msg)

    # --- HANDLE QUESTION ---
    if question:
        st.session_state.chat.append(("user", question))

        with chat_area:
            with st.chat_message("user", avatar="🧑"):
                st.write(question)

            with st.chat_message("assistant", avatar="✨"):
                with st.spinner("Thinking..."):
                    if mode == "Ask about AI news (RAG)":
                        result = rag_answer(question, top_k=5)
                        answer = result.get("answer", "")
                        sources = result.get("sources", [])

                        st.write(answer if answer else "I couldn't find an answer in the news context.")

                        if sources:
                            st.caption("Sources:")
                            for s in sources[:5]:
                                st.write(f"- {s.get('title','')} ({s.get('published','')}) — {s.get('url','')}")
                    else:
                        answer = chat_ai(question)
                        st.write(answer)

        st.session_state.chat.append(("assistant", answer))