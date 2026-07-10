import streamlit as st
from datetime import datetime
import os
import requests

st.set_page_config(page_title="Kids Activity Runner", layout="wide")

#try:
    #API_BASE_URL = st.secrets["API_BASE_URL"]
#except Exception:

    #API_BASE_URL = os.getenv(
       # "API_BASE_URL",
        #"https://kids-activity-runner-api1.onrender.com"
    #)
API_BASE_URL = "https://kids-activity-runner-v6t7.onrender.com"


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
        padding-top: 2.5rem !important;
      }
      [data-testid="stHeader"] {
        background: white !important;
        border-bottom: none;
      }

      [data-testid="stAppViewContainer"] {
        background: white;
      }

.block-container {
    padding-top: 0.5rem;
}
      .activity-card {
          background: #FFFFFF;
          border-radius: 14px;
          padding: 14px 16px;
          margin-bottom: 14px;
          border: 1px solid #E5E7EB;
          box-shadow: 0 4px 12px rgba(0,0,0,0.06);
      }

      h1 { color: #2C2C54 !important; }

      .hero-title {
        text-align: center;
        margin-top: 1rem;
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
            timeout=25,
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
            timeout=25,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("answer", {})
    except requests.exceptions.RequestException as e:
        return f"Backend error while answering question: {e}"


if "chat" not in st.session_state:
    st.session_state.chat = []

left_header, center_header, right_header = st.columns([1.5, 4.5, 1])

with left_header:
    image_path = os.path.join(os.path.dirname(__file__), "assets", "kiddo.png")
    if os.path.exists(image_path):
        st.markdown("<div style='padding-left:40px;'>", unsafe_allow_html=True)
        st.image(image_path, width=180)
        st.markdown("</div>", unsafe_allow_html=True)

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

activities = data.get("articles", [])[:2]

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
    st.markdown("<div class='demo-label'>Enter your city: </div>", unsafe_allow_html=True)
    location = st.text_input("Enter your city", value="Sunnyvale", label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='demo-label'>Try one of theese:</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    d1, d2, d3, d4 = st.columns(4)

    question = None
    if d1.button("🎨 Art classes"):
        question = f"kids art classes near {location}"
        display_text = "Are there any art classes for kids nearby?"

    elif d2.button("🌄 Outdoor fun"):
        question = f"kids parks playgrounds near {location}"
        display_text = "Are there any park playgrounds for kids nearby?"

    elif d3.button("🖼 Museums"):
        question = f"children's museums near {location}"
        display_text = "Are there any children's museums for kids nearby?"

    elif d4.button("🎪 Weekend events"):
        question = f"family events this weekend near {location}"
        display_text = "Are there any family events this weekend nearby?"
    st.divider()

    chat_area = st.container()
    #if st.button("Clear chat"):
        #st.session_state.chat = []
        #st.rerun()

    with chat_area:
        for role, msg in st.session_state.chat:
            avatar = "🧑" if role == "user" else "✨"
            with st.chat_message(role, avatar=avatar):
                if role == "assistant" and isinstance(msg, dict) and "results" in msg:
                    for place in msg["results"]:
                        st.subheader(place.get("name", ""))
                        st.write(place.get("reason", ""))
                        website = place.get("website", "")
                        if website:
                            st.markdown(f"[Visit Website]({website})")
                        else:
                            st.caption("Website not available")
                else:
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
        question = f"{typed_question} near {location}"

    if question:
        st.session_state.chat.append(("user", question))

        with st.chat_message("assistant", avatar="✨"):
            with st.spinner("Thinking..."):
                answer = ask_general_question(question)
                if isinstance(answer, dict) and "results" in answer:
                    for place in answer["results"]:
                        st.subheader(place.get("name", ""))
                        reason = place.get("reason", "")
                        if reason:
                            st.write(reason)

                        website = place.get("website", "")
                        if website:
                            label = "Visit Website"
                            if "google.com/maps" in website or "maps.google" in website:
                                label = "View on Map"

                            st.markdown(f"[{label}]({website})")
                        else:
                            st.caption("Official website not available")

                else:
                    st.write(answer)     

        if isinstance(answer, dict) and "results" in answer:
            st.session_state.chat.append(("assistant", answer))
        else:
            st.session_state.chat.append(("assistant", str(answer)))
        st.rerun()