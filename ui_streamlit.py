import streamlit as st
from datetime import datetime
import os
import requests

st.set_page_config(page_title="Kids Activity Runner", layout="wide")

try:
    API_BASE_URL = st.secrets["API_BASE_URL"]
except Exception:
    API_BASE_URL = os.getenv(
        "API_BASE_URL",
        "https://kids-activity-runner-api1.onrender.com"
    )

st.markdown(
    """
    <style>
      [data-testid="stSidebar"] {display: none;}
      [data-testid="stSidebarNav"] {display: none;}
      [data-testid="stToolbar"] {display: none;}
      footer {visibility: hidden;}
      .stDeployButton {display:none;}
      .viewerBadge_container__1QSob {display:none;}

      .center-title {
        text-align: center;
        margin-top: -15px;
      }

      .block-container {
        padding-top: 0.5rem !important;
      }

      [data-testid="stAppViewContainer"] {
        background: #B7C9C9;
      }

      [data-testid="stHeader"] {
        background: transparent !important;
      }

      h1 { color: #2C2C54 !important; }

      [data-testid="stRadio"] > div {
        background: #F1F3F6 !important;
        padding: 8px 12px !important;
        border-radius: 12px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_activity_data(limit=2, include_summary=True):
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/news",
            params={
                "limit": limit,
                "include_summary": include_summary,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch activity data from backend: {e}")
        return {"articles": [], "summary": ""}


def ask_general_question(question: str):
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/chat",
            json={"question": question},
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("answer", "No answer returned from backend.")
    except requests.exceptions.RequestException as e:
        return f"Backend error while answering general question: {e}"


def ask_activity_rag(question: str, top_k: int = 5):
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/rag/answer",
            json={
                "question": question,
                "top_k": top_k,
            },
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {
            "answer": f"Backend error while answering activity question: {e}",
            "sources": [],
        }


if "chat" not in st.session_state:
    st.session_state.chat = []

LOCATION_OPTIONS = [
    "Sunnyvale",
    "San Jose",
    "Mountain View",
    "Palo Alto",
    "Santa Clara",
    "Bay Area"
]

left_header, center_header, right_header = st.columns([1, 5, 1])

with left_header:
    image_path = os.path.join(os.path.dirname(__file__), "assets", "kiddo.png")
    st.image(image_path, width=290)

with center_header:
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0px;'>Kids Activity Runner</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center; font-size:18px; color:#444; margin-top:4px;'>Find family-friendly activities near you using AI</p>",
        unsafe_allow_html=True
    )

    today = datetime.now().strftime("%b %d %Y")

    st.markdown(
        f"<p style='text-align:center; color:#666; margin-top:0px;'>{today}</p>",
        unsafe_allow_html=True
    )

with st.spinner("Fetching activity ideas..."):
    data = fetch_activity_data(limit=2, include_summary=True)

activities = data.get("articles", [])[:2]
summary_text = data.get("summary", "")

left, right = st.columns([1, 3], gap="large")

with left:
    st.subheader("📍 Activity Summary")
    st.write(summary_text if summary_text else "No summary available.")

    st.subheader("📰 Suggested Activities")
    if not activities:
        st.warning("No activities returned.")
    else:
        for a in activities:
            with st.container(border=True):
                st.markdown(f"**{a.get('title', 'Untitled')}**")
                if a.get("published"):
                    st.caption(a["published"])
                if a.get("url"):
                    st.link_button("View activity", a["url"])

with right:
    selected_location = st.selectbox("Choose area", LOCATION_OPTIONS, index=0)

    mode = st.radio(
        "Chat mode",
        ["General question", "Ask about kids activities (RAG)"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("**Demo questions:**")
    d1, d2, d3 = st.columns(3)

    if d1.button("Weekend activities"):
        question = "What family activities are happening this weekend?"
    elif d2.button("Kids museums"):
        question = "Are there kids museums or science centers nearby?"
    elif d3.button("Outdoor parks"):
        question = "What outdoor parks are good for kids nearby?"
    else:
        question = None

    st.divider()

    chat_area = st.container()

    with chat_area:
        for role, msg in st.session_state.chat:
            avatar = "🧑" if role == "user" else "✨"
            with st.chat_message(role, avatar=avatar):
                st.write(msg)

    st.markdown(
        """
        <script>
            const chatMessages = window.parent.document.querySelectorAll('[data-testid="stChatMessage"]');
            if (chatMessages.length > 0) {
                chatMessages[chatMessages.length - 1].scrollIntoView({ behavior: "smooth", block: "start" });
            }
        </script>
        """,
        unsafe_allow_html=True,
    )

    typed_question = st.chat_input("Ask about kids activities near your area...")

    if typed_question:
        question = typed_question

    if question:
        question_with_location = f"{question} Default location: {selected_location}."
        st.session_state.chat.append(("user", question))

        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Thinking..."):
                if mode == "Ask about kids activities (RAG)":
                    result = ask_activity_rag(question_with_location, top_k=5)
                    answer = result.get("answer", "")
                    sources = result.get("sources", [])

                    st.write(answer if answer else "I couldn't find relevant activity information.")

                    if sources:
                        st.caption("Activity Sources:")
                        for s in sources[:5]:
                            with st.container(border=True):
                                st.markdown(f"**{s.get('title', 'Untitled')}**")
                                if s.get("published"):
                                    st.caption(s["published"])
                                if s.get("url"):
                                    st.link_button("Open source", s["url"])
                else:
                    answer = ask_general_question(question_with_location)
                    st.write(answer)

        st.session_state.chat.append(("assistant", answer))
        st.rerun()