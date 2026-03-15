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

      .block-container {
        padding-top: 0.8rem !important;
      }

      [data-testid="stAppViewContainer"] {
        background: #B7C9C9;
      }

      [data-testid="stHeader"] {
        background: transparent !important;
      }

      h1 { color: #2C2C54 !important; }

      .hero-title {
        text-align: center;
        margin-bottom: 0.2rem;
      }

      .hero-subtitle {
        text-align: center;
        font-size: 18px;
        color: #444;
        margin-top: 0;
        margin-bottom: 0.4rem;
      }

      .hero-date {
        text-align: center;
        color: #666;
        margin-top: 0;
        margin-bottom: 1.2rem;
      }

      .demo-label {
        font-weight: 600;
        margin-top: 0.4rem;
        margin-bottom: 0.5rem;
      }
      .activity-card {
    background: white;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 14px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}

.activity-title {
    font-weight: 600;
    font-size: 16px;
    color: #2C2C54;
    margin-bottom: 4px;
}

.activity-date {
    font-size: 12px;
    color: #666;
}
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_activity_data(limit=4, include_summary=True):
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
        return f"Backend error while answering question: {e}"


if "chat" not in st.session_state:
    st.session_state.chat = []

left_header, center_header, right_header = st.columns([1, 5, 1])

with left_header:
    image_path = os.path.join(os.path.dirname(__file__), "assets", "kiddo.png")
    if os.path.exists(image_path):
        st.image(image_path, width=220)

with center_header:
    st.markdown(
        "<h1 class='hero-title'>Kids Activity Runner</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p class='hero-subtitle'>Find family-friendly activities near you using AI</p>",
        unsafe_allow_html=True
    )

    today = datetime.now().strftime("%b %d %Y")
    st.markdown(
        f"<p class='hero-date'>{today}</p>",
        unsafe_allow_html=True
    )

with st.spinner("Fetching activity ideas..."):
    data = fetch_activity_data(limit=4, include_summary=True)

activities = data.get("articles", [])[:4]

left, right = st.columns([1, 2.2], gap="large")

with left:
    st.subheader("📰 Suggested Activities")

    if not activities:
        st.warning("No activities returned.")
    else:
       for a in activities:

           title = a.get("title", "Untitled")
           date = a.get("published", "")
           url = a.get("url", "")

           st.markdown(
               f"""
               <div class="activity-card">
                   <div class="activity-title">{title}</div>
                   <div class="activity-date">{date}</div>
               </div>
               """,
               unsafe_allow_html=True,
            )

           if url:
               st.link_button("Open Activity", url)

with right:
    st.markdown("<div class='demo-label'>Try one of these:</div>", unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)

    question = None
    if d1.button("🎨 Art classes"):
        question = "Are there any art classes, painting sessions, or pottery activities for kids nearby?"
    elif d2.button("🏞 Outdoor fun"):
        question = "What outdoor activities or parks are good for kids nearby?"
    elif d3.button("🧪 Museums"):
        question = "Are there any kids museums or science centers nearby?"
    elif d4.button("🎪 Weekend events"):
        question = "What family-friendly events are happening this weekend nearby?"

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

    typed_question = st.chat_input("Ask about kids activities near you...")

    if typed_question:
        question = typed_question

    if question:
        st.session_state.chat.append(("user", question))

        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Thinking..."):
                answer = ask_general_question(question)
                st.write(answer)

        st.session_state.chat.append(("assistant", answer))
        st.rerun()