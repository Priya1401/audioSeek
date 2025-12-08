import streamlit as st
import requests
import os
import uuid
from dotenv import load_dotenv
import base64
import json
import time
from google_auth_oauthlib.flow import Flow

# Load environment variables
load_dotenv(".env.local")

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8001")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
GCS_IMAGES_BASE_URL = os.getenv("GCS_IMAGES_BASE_URL", "https://storage.googleapis.com/your-bucket-name/images")

st.set_page_config(page_title="AudioSeek", layout="wide", initial_sidebar_state="expanded")

# ========================================================================
# PURE BLACK THEME CSS - NO EMOJIS
# ========================================================================
st.markdown("""
    <style>
        /* Pure Black Background */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
            background: #000000 !important;
        }
        
        /* Sidebar - Pure Black */
        [data-testid="stSidebar"] {
            background: #000000 !important;
            border-right: 2px solid #00d9ff;
        }
        
        [data-testid="stSidebarContent"] {
            background: transparent !important;
        }
        
        /* Main Content */
        .stMainBlockContainer {
            background: transparent !important;
        }
        
        /* Typography - White on Black */
        h1 {
            color: #00d9ff !important;
            font-weight: 800 !important;
            font-size: 3.5rem !important;
            letter-spacing: -1px;
        }
        
        h2 {
            color: #ffffff !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        h3 {
            color: #00d9ff !important;
            font-weight: 600 !important;
        }
        
        p, span, label, div {
            color: #ffffff !important;
        }
        
        /* Book Cards - Pure Black */
        .book-card {
            position: relative;
            background: #000000;
            border: 2px solid #00d9ff;
            border-radius: 16px;
            padding: 20px;
            transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
            box-shadow: 0 0 20px rgba(0, 217, 255, 0.3);
            overflow: hidden;
        }
        
        .book-card:hover {
            border-color: #00ffff;
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 15px 40px rgba(0, 217, 255, 0.5);
        }
        
        .book-cover-image {
            width: 100%;
            height: 320px;
            border-radius: 12px;
            object-fit: cover;
            box-shadow: 0 8px 24px rgba(0, 217, 255, 0.3);
            transition: all 0.4s ease;
            border: 2px solid #00d9ff;
        }
        
        .book-card:hover .book-cover-image {
            box-shadow: 0 12px 40px rgba(0, 217, 255, 0.5);
            border-color: #00ffff;
        }
        
        .book-cover-placeholder {
            width: 100%;
            height: 320px;
            background: #0a0a0a;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #00d9ff;
            font-weight: 700;
            font-size: 18px;
            text-align: center;
            padding: 20px;
            border: 2px dashed #00d9ff;
        }
        
        .book-info {
            margin: 16px 0;
            position: relative;
            z-index: 1;
        }
        
        .book-title {
            font-size: 18px;
            font-weight: 700;
            color: #ffffff !important;
            margin: 12px 0 8px 0;
            line-height: 1.3;
        }
        
        .book-author {
            font-size: 13px;
            color: #a8b0c1 !important;
            margin: 0;
            font-style: italic;
        }
        
        /* Sidebar Book Thumbnail */
        .sidebar-book-thumbnail {
            width: 100%;
            max-width: 200px;
            border-radius: 8px;
            border: 2px solid #00d9ff;
            box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
            margin: 10px auto;
            display: block;
        }
        
        /* Chat sidebar thumbnail - smaller */
        .chat-book-thumbnail {
            width: 150px;
            border-radius: 8px;
            border: 2px solid #00d9ff;
            box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
            margin: 10px 0;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%) !important;
            color: #000000 !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            font-size: 14px !important;
            padding: 12px 24px !important;
            transition: all 0.3s ease !important;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(0, 217, 255, 0.6) !important;
            background: linear-gradient(135deg, #00ffff 0%, #00d9ff 100%) !important;
        }
        
        /* Secondary button style */
        .stButton > button[kind="secondary"] {
            background: transparent !important;
            border: 2px solid #00d9ff !important;
            color: #00d9ff !important;
        }
        
        .stButton > button[kind="secondary"]:hover {
            background: rgba(0, 217, 255, 0.1) !important;
            border-color: #00ffff !important;
            color: #00ffff !important;
        }
        
        /* Input Fields - Pure Black */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input {
            background-color: #000000 !important;
            border: 2px solid #333333 !important;
            color: #ffffff !important;
            border-radius: 10px !important;
            padding: 12px 16px !important;
            font-size: 14px !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus {
            border-color: #00d9ff !important;
            box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.2) !important;
            background-color: #000000 !important;
        }
        
        /* Chat Input - Pure Black */
        .stChatInputContainer {
            background: #000000 !important;
            border-top: 2px solid #00d9ff;
            padding: 20px !important;
        }
        
        /* Chat Messages */
        .chat-message-user {
            background: linear-gradient(135deg, #00d9ff 0%, #00b8cc 100%);
            color: #000000;
            border-radius: 16px;
            padding: 16px 20px;
            margin: 12px 0;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(0, 217, 255, 0.4);
            max-width: 75%;
            margin-left: auto;
            word-wrap: break-word;
            font-size: 15px;
            line-height: 1.5;
        }
        
        .chat-message-assistant {
            background: #000000;
            color: #ffffff;
            border: 2px solid #00d9ff;
            border-radius: 16px;
            padding: 16px 20px;
            margin: 12px 0;
            box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
            max-width: 85%;
            word-wrap: break-word;
            font-size: 15px;
            line-height: 1.6;
        }
        
        /* Compact empty state - Pure Black */
        .chat-empty-compact {
            padding: 30px 20px;
            text-align: center;
            background: #000000;
            border-radius: 12px;
            border: 2px dashed #00d9ff;
            margin: 20px 0;
        }
        
        /* Sidebar Navigation */
        [data-testid="stRadio"] label {
            color: #ffffff !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            font-size: 15px !important;
        }
        
        [data-testid="stRadio"] input {
            accent-color: #00d9ff !important;
        }
        
        /* Divider */
        hr {
            border-color: #00d9ff !important;
            opacity: 0.5;
        }
        
        /* Status Messages - Pure Black */
        .stSuccess {
            background-color: #000000 !important;
            color: #4ade80 !important;
            border: 2px solid #4ade80 !important;
            border-radius: 8px !important;
        }
        
        .stError {
            background-color: #000000 !important;
            color: #f87171 !important;
            border: 2px solid #f87171 !important;
            border-radius: 8px !important;
        }
        
        .stInfo {
            background-color: #000000 !important;
            color: #00d9ff !important;
            border: 2px solid #00d9ff !important;
            border-radius: 8px !important;
        }
        
        .stWarning {
            background-color: #000000 !important;
            color: #fbbf24 !important;
            border: 2px solid #fbbf24 !important;
            border-radius: 8px !important;
        }
        
        /* Modal/Dialog - Pure Black */
        .stDialog {
            background: #000000 !important;
            border: 2px solid #00d9ff !important;
            border-radius: 16px !important;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: #000000 !important;
            border-bottom: 2px solid #00d9ff !important;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #ffffff !important;
            background-color: #000000 !important;
        }
        
        .stTabs [aria-selected="true"] {
            color: #00d9ff !important;
            border-bottom-color: #00d9ff !important;
        }
        
        /* Progress bar */
        .stProgress > div > div > div > div {
            background-color: #00d9ff !important;
        }
        
        /* Expander */
        .streamlit-expanderHeader {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 2px solid #333333 !important;
            border-radius: 8px !important;
        }
        
        .streamlit-expanderHeader:hover {
            border-color: #00d9ff !important;
        }
        
        /* Container borders */
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
            border-color: #333333 !important;
        }
        
        /* Caption text */
        .stCaptionContainer {
            color: #a8b0c1 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ========================================================================
# SESSION STATE INITIALIZATION
# ========================================================================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_book" not in st.session_state:
    st.session_state.selected_book = None

if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "user_name" not in st.session_state:
    st.session_state.user_name = None

if "auth_token" not in st.session_state:
    st.session_state.auth_token = None

if "until_chapter" not in st.session_state:
    st.session_state.until_chapter = 0
    
if "until_time_total" not in st.session_state:
    st.session_state.until_time_total = 0

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================

def get_book_image_url(book_id):
    """Get book cover image URL from GCS or local fallback"""
    gcs_url = f"{GCS_IMAGES_BASE_URL}/{book_id}.png"
    local_path = f"./images/{book_id}.png"
    return local_path  # Change to gcs_url when GCS is configured

def fetch_books_from_api():
    """Fetch books consistently from API"""
    try:
        response = requests.get(f"{API_URL}/books", timeout=5)
        if response.status_code == 200:
            books = response.json()
            return books if books else []
    except Exception as e:
        st.error(f"Failed to fetch books: {e}")
    return []

# ========================================================================
# AUTHENTICATION - GOOGLE OAUTH (PERSISTENT)
# ========================================================================
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div style='text-align: center; padding: 60px 0;'>", unsafe_allow_html=True)
        st.markdown("<h1 style='font-size: 56px; margin-bottom: 10px;'>AudioSeek</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #a8b0c1; font-size: 18px; margin-bottom: 50px; letter-spacing: 2px;'>DISCOVER AUDIOBOOK INSIGHTS POWERED BY AI</p>", unsafe_allow_html=True)
        
        if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and GOOGLE_REDIRECT_URI:
            try:
                client_config = {
                    "installed": {
                        "client_id": GOOGLE_CLIENT_ID,
                        "client_secret": GOOGLE_CLIENT_SECRET,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [GOOGLE_REDIRECT_URI]
                    }
                }
                
                flow = Flow.from_client_config(
                    client_config,
                    scopes=["https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile", "openid"]
                )
                flow.redirect_uri = GOOGLE_REDIRECT_URI
                
                query_params = st.query_params
                
                if "code" in query_params and not st.session_state.authenticated:
                    auth_code = query_params["code"]
                    try:
                        flow.fetch_token(code=auth_code)
                        credentials = flow.credentials
                        
                        id_token = credentials.id_token
                        if id_token:
                            parts = id_token.split(".")
                            if len(parts) > 1:
                                payload = parts[1]
                                payload += "=" * ((4 - len(payload) % 4) % 4)
                                decoded = base64.urlsafe_b64decode(payload)
                                user_info = json.loads(decoded)
                                st.session_state.user_email = user_info.get("email")
                                st.session_state.user_name = user_info.get("name")
                                st.session_state.auth_token = id_token
                                st.session_state.authenticated = True
                                st.success(f"Welcome, {st.session_state.user_name}!")
                                time.sleep(1)
                                st.query_params.clear()
                                st.rerun()
                    except Exception as e:
                        st.error(f"Authentication failed: {e}")
                else:
                    auth_url, state = flow.authorization_url()
                    
                    st.markdown(f"""
                        <div style='text-align: center; margin-top: 40px;'>
                            <a href='{auth_url}' style='
                                display: inline-block;
                                background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
                                color: #000000;
                                padding: 16px 48px;
                                border-radius: 12px;
                                text-decoration: none;
                                font-weight: 700;
                                font-size: 16px;
                                letter-spacing: 2px;
                                text-transform: uppercase;
                                box-shadow: 0 8px 25px rgba(0, 217, 255, 0.3);
                                transition: all 0.3s ease;
                            ' onmouseover="this.style.transform='translateY(-3px)'; this.style.boxShadow='0 12px 35px rgba(0, 217, 255, 0.5)'"
                               onmouseout="this.style.transform='none'; this.style.boxShadow='0 8px 25px rgba(0, 217, 255, 0.3)'">
                                Sign in with Google
                            </a>
                        </div>
                    """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Authentication error: {e}")
        else:
            st.warning("Google OAuth credentials not configured in .env.local")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.stop()

# ========================================================================
# MAIN APPLICATION (Only reachable if authenticated)
# ========================================================================

# Determine current page based on navigation
if "current_page" not in st.session_state:
    st.session_state.current_page = "Library"

# Sidebar
with st.sidebar:
    st.markdown(f"<p style='color: #00d9ff; font-weight: 700; font-size: 16px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 20px;'>Welcome, {st.session_state.user_name}</p>", unsafe_allow_html=True)
    
    if st.button("Sign Out", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.divider()
    
    # Back button when in Chat page
    if st.session_state.current_page == "Chat" and st.session_state.selected_book:
        if st.button("← Back to Library", use_container_width=True, type="secondary"):
            st.session_state.current_page = "Library"
            st.session_state.selected_book = None
            st.session_state.messages = []
            st.rerun()
        st.divider()
    
    # Show selected book thumbnail in sidebar when chatting
    if st.session_state.selected_book and st.session_state.current_page == "Chat":
        st.markdown("### Currently Reading")
        book = st.session_state.selected_book
        try:
            st.image(get_book_image_url(book['book_id']), use_container_width=True)
        except:
            st.markdown(f"<div style='text-align: center; padding: 20px; background: #0a0a0a; border: 2px dashed #00d9ff; border-radius: 8px;'><p style='color: #00d9ff;'>{book.get('title', 'Selected Book')}</p></div>", unsafe_allow_html=True)
        st.divider()
    
    st.header("Navigation")
    
    # Navigation options
    if st.session_state.selected_book and st.session_state.current_page == "Chat":
        page_options = ["Chat", "Library", "My Activity", "Add New Book", "Health Check"]
    else:
        page_options = ["Library", "My Activity", "Add New Book", "Health Check"]
    
    selected_page = st.radio("Go to", page_options, index=page_options.index(st.session_state.current_page) if st.session_state.current_page in page_options else 0)
    
    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()

page = st.session_state.current_page

# ========================================================================
# PAGE: CHAT (Dedicated full-screen chat)
# ========================================================================
if page == "Chat" and st.session_state.selected_book:
    book = st.session_state.selected_book
    
    # Sidebar adjustments for Chat
    with st.sidebar:
        st.divider()
        st.subheader("Spoiler Control")
        st.caption("Restrict answers to your reading progress")
        
        # Current values
        current_chapter = st.session_state.until_chapter
        current_time = st.session_state.until_time_total
        
        new_chapter = st.number_input(
            "Until Chapter", 
            min_value=0, 
            value=current_chapter, 
            help="0 = search all chapters",
            key="chapter_input"
        )
        
        c1, c2 = st.columns(2)
        with c1:
            new_minutes = st.number_input(
                "Minutes", 
                min_value=0, 
                value=current_time // 60, 
                step=1,
                key="minutes_input"
            )
        with c2:
            new_seconds = st.number_input(
                "Seconds", 
                min_value=0, 
                value=current_time % 60, 
                step=1,
                key="seconds_input"
            )
        
        new_time_total = (new_minutes * 60) + new_seconds
        
        # Check if values changed
        if new_chapter != current_chapter or new_time_total != current_time:
            st.warning("Changing progress will reset your chat session!")
            if st.button("Apply Changes & Reset Session", type="primary", use_container_width=True):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.session_state.until_chapter = new_chapter
                st.session_state.until_time_total = new_time_total
                st.success("Session reset! Progress updated.")
                time.sleep(1)
                st.rerun()
        
        until_chapter = st.session_state.until_chapter
        until_time_total = st.session_state.until_time_total

    # Clean title display - no "Unknown (GCS)"
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        title = book.get('title', 'Untitled')
        st.markdown(f"<h2 style='text-align: center; margin-bottom: 5px;'>{title}</h2>", unsafe_allow_html=True)
        author = book.get('author', '')
        if author and author not in ['Unknown', 'Unknown Author', '']:
            st.markdown(f"<p style='text-align: center; color: #a8b0c1; font-size: 16px; margin: 0;'>{author}</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # Compact empty state that disappears after first message
    if st.session_state.messages:
        # Show full chat history with book thumbnail
        col_chat, col_thumb = st.columns([4, 1])
        
        with col_chat:
            st.markdown("<div style='height: 500px; overflow-y: auto; padding: 24px; background: #000000; border-radius: 12px; border: 2px solid #00d9ff;'>", unsafe_allow_html=True)
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message-user">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message-assistant">{message["content"]}</div>', unsafe_allow_html=True)
                    if "audio" in message:
                        for ref in message["audio"]:
                            if "url" in ref:
                                start_time = int(ref.get("start_time", 0))
                                st.audio(ref["url"], start_time=start_time)
                                st.caption(f"Chapter {ref.get('chapter_id')} at {start_time}s")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col_thumb:
            # Book thumbnail on the side
            try:
                st.markdown("<div style='position: sticky; top: 20px;'>", unsafe_allow_html=True)
                st.image(get_book_image_url(book['book_id']), caption=book.get('title'), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            except:
                pass
    else:
        # Compact empty state
        st.markdown("<div class='chat-empty-compact'><p style='color: #a8b0c1; font-size: 16px; margin: 0;'>Start asking questions about this book</p></div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Chat input
    prompt = st.chat_input("Ask a question about this audiobook...", key="chat_input")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "query": prompt,
                    "book_id": book['book_id'],
                    "session_id": st.session_state.session_id,
                    "user_email": st.session_state.user_email,
                    "until_chapter": int(until_chapter) if until_chapter > 0 else None,
                    "until_time_seconds": float(until_time_total) if until_time_total > 0 else None
                }
                response = requests.post(f"{API_URL}/qa/ask", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get("answer", "No answer provided.")
                    
                    msg = {"role": "assistant", "content": answer}
                    if "audio_references" in result:
                         msg["audio"] = result["audio_references"]
                    
                    st.session_state.messages.append(msg)
                else:
                    error_msg = f"Error: {response.status_code}"
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    st.error(error_msg)
            except Exception as e:
                error_msg = f"Connection failed: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                st.error(error_msg)
        
        st.rerun()

# ========================================================================
# PAGE: LIBRARY
# ========================================================================
elif page == "Library":
    st.header("Audiobook Library")
    
    # Refresh button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Refresh Library", use_container_width=True):
            st.rerun()
    
    books = fetch_books_from_api()
    
    if not books:
        st.info("No books found in the library. Try refreshing or add a new book.")
    else:
        cols = st.columns(2, gap="large")
        for i, book in enumerate(books):
            with cols[i % 2]:
                st.markdown('<div class="book-card">', unsafe_allow_html=True)
                
                # Try to load image
                try:
                    st.image(get_book_image_url(book['book_id']), use_container_width=True)
                except:
                    st.markdown('<div class="book-cover-placeholder">Book Cover</div>', unsafe_allow_html=True)
                
                st.markdown(f'<div class="book-info"><p class="book-title">{book.get("title", "Untitled")}</p><p class="book-author">{book.get("author", "Unknown Author")}</p></div>', unsafe_allow_html=True)
                
                # Force refresh by using unique key
                if st.button("Start Chat", key=f"chat_{book['book_id']}_{i}_{time.time()}", use_container_width=True):
                    st.session_state.selected_book = book
                    st.session_state.messages = []
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.until_chapter = 0
                    st.session_state.until_time_total = 0
                    st.session_state.current_page = "Chat"
                    time.sleep(0.1)
                    st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)

elif page == "Add New Book":
    st.header("Request New Book")
    st.write("Upload an audio file (MP3/WAV) or a ZIP file containing audio chapters.")
    
    with st.form("upload_form"):
        book_name = st.text_input("Book Name")
        uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "zip"])
        submitted = st.form_submit_button("Upload Book")
        
        if submitted:
            if not book_name:
                st.error("Please enter a book name.")
            elif not uploaded_file:
                st.error("Please upload a file.")
            else:
                processed_book_name = book_name.strip().lower().replace(" ", "_")
                
                with st.spinner("Uploading..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        data = {"book_name": processed_book_name}
                        
                        response = requests.post(f"{API_URL}/upload-audio", files=files, data=data)
                        
                        if response.status_code == 200:
                            upload_result = response.json()
                            st.success(f"Successfully uploaded '{book_name}'!")
                            
                            st.info("Submitting processing job...")
                            
                            try:
                                process_payload = {
                                    "folder_path": upload_result["folder_path"],
                                    "book_name": upload_result["book_name"],
                                    "target_tokens": 512,
                                    "overlap_tokens": 50,
                                    "model_size": "base",
                                    "beam_size": 5,
                                    "compute_type": "float32",
                                    "user_email": st.session_state.user_email
                                }
                                
                                process_response = requests.post(f"{API_URL}/process-audio", json=process_payload)
                                
                                if process_response.status_code == 200:
                                    job_info = process_response.json()
                                    st.success(f"Job Submitted! ID: {job_info.get('job_id')}")
                                    st.info("You'll receive an email when processing is complete!")
                                else:
                                    st.error(f"Submission failed: {process_response.status_code}")
                                    
                            except Exception as e:
                                st.error(f"Processing connection error: {e}")
                        else:
                            st.error(f"Upload failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

# ========================================================================
# PAGE: MY ACTIVITY
# ========================================================================
elif page == "My Activity":
    st.header("My Activity")
    
    tab1, tab2 = st.tabs(["Upload History", "Chat History"])
    
    with tab1:
        st.subheader("Your Upload Jobs")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("Refresh Status", use_container_width=True):
                st.rerun()
        
        try:
            response = requests.get(f"{API_URL}/jobs/user/{st.session_state.user_email}")
            if response.status_code == 200:
                jobs = response.json()
                if not jobs:
                    st.info("No uploads yet. Go to 'Add New Book' to upload an audiobook.")
                else:
                    for job in jobs:
                        with st.container(border=True):
                            col1, col2, col3 = st.columns([2, 1, 1])
                            
                            with col1:
                                st.markdown(f"<p style='font-size: 18px; font-weight: 700; color: #ffffff; margin: 0;'>{job.get('book_name', 'Unknown Book')}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='font-size: 13px; color: #a8b0c1; margin: 5px 0 0 0;'>Job ID: <code>{job.get('job_id', 'N/A')}</code></p>", unsafe_allow_html=True)
                                if job.get('created_at'):
                                    st.markdown(f"<p style='font-size: 12px; color: #a8b0c1;'>Started: {job.get('created_at')}</p>", unsafe_allow_html=True)
                            
                            with col2:
                                status = job.get('status', 'unknown')
                                if status == 'processing':
                                    st.markdown("<span style='background: #000000; color: #fbbf24; padding: 6px 12px; border: 2px solid #fbbf24; border-radius: 6px; font-weight: 600; font-size: 12px;'>PROCESSING</span>", unsafe_allow_html=True)
                                elif status == 'completed':
                                    st.markdown("<span style='background: #000000; color: #4ade80; padding: 6px 12px; border: 2px solid #4ade80; border-radius: 6px; font-weight: 600; font-size: 12px;'>COMPLETED</span>", unsafe_allow_html=True)
                                elif status == 'failed':
                                    st.markdown("<span style='background: #000000; color: #f87171; padding: 6px 12px; border: 2px solid #f87171; border-radius: 6px; font-weight: 600; font-size: 12px;'>FAILED</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown(f"<span style='background: #000000; color: #a8b0c1; padding: 6px 12px; border: 2px solid #a8b0c1; border-radius: 6px; font-weight: 600; font-size: 12px;'>{status.upper()}</span>", unsafe_allow_html=True)
                            
                            with col3:
                                if status == 'processing':
                                    progress = job.get('progress', 0.0)
                                    st.progress(progress)
                                    st.caption(f"{int(progress * 100)}%")
                            
                            if job.get('message'):
                                st.markdown(f"<p style='color: #a8b0c1; font-size: 12px; margin: 8px 0 0 0;'>{job.get('message')}</p>", unsafe_allow_html=True)
                            
                            if status == 'completed' and job.get('completed_at'):
                                st.markdown(f"<p style='color: #4ade80; font-size: 12px;'>Completed: {job.get('completed_at')}</p>", unsafe_allow_html=True)
            else:
                st.error(f"Failed to fetch upload history: {response.status_code}")
        except Exception as e:
            st.warning(f"Upload history service unavailable: {e}")
    
    with tab2:
        st.subheader("Your Chat History")
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            if st.button("Refresh Chats", use_container_width=True):
                st.rerun()
        
        try:
            response = requests.get(f"{API_URL}/chat-history/{st.session_state.user_email}")
            if response.status_code == 200:
                chat_history = response.json()
                if not chat_history:
                    st.info("No chat history yet. Start a conversation with a book!")
                else:
                    for idx, chat in enumerate(chat_history):
                        with st.container(border=True):
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                book_title = chat.get('book_title', 'Unknown Book')
                                st.markdown(f"<p style='font-size: 16px; font-weight: 700; color: #00d9ff;'>{book_title}</p>", unsafe_allow_html=True)
                            
                            with col2:
                                timestamp = chat.get('timestamp', 'N/A')
                                st.markdown(f"<p style='color: #a8b0c1; font-size: 11px; text-align: right;'>{timestamp}</p>", unsafe_allow_html=True)
                            
                            question = chat.get('question', '')
                            answer = chat.get('answer', '')
                            
                            st.markdown(f"<p style='color: #ffffff; margin: 10px 0 5px 0;'><strong>You:</strong> {question}</p>", unsafe_allow_html=True)
                            
                            if len(answer) > 200:
                                with st.expander("View Answer"):
                                    st.markdown(f"<p style='color: #a8b0c1;'>{answer}</p>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<p style='color: #a8b0c1; margin: 0;'><strong>AudioSeek:</strong> {answer}</p>", unsafe_allow_html=True)
            else:
                st.warning("Chat history service not yet available")
        except Exception as e:
            st.info("Chat history feature coming soon!")

# ========================================================================
# PAGE: HEALTH CHECK
# ========================================================================
elif page == "Health Check":
    st.header("Service Health")
    
    if st.button("Check Health"):
        try:
            response = requests.get(f"{API_URL}/health")
            st.json(response.json())
            st.success("Backend service is healthy!")
        except Exception as e:
            st.error(f"Failed to connect to service: {e}")