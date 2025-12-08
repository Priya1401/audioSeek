import streamlit as st
import requests
import os
import uuid
from dotenv import load_dotenv
import base64
import json
import time
import streamlit.components.v1 as components
import altair as alt
import pandas as pd
from google_auth_oauthlib.flow import Flow
from google.cloud import secretmanager

# Load environment variables
load_dotenv(".env.local")

# Configuration
# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8001")
PROJECT_ID = os.getenv("PROJECT_ID")

def get_secret(secret_id, default=None):
    """
    Tries to get secret from:
    1. Environment Variable
    2. GCP Secret Manager (if PROJECT_ID is set)
    3. Default value
    """
    # 1. Env Var
    val = os.getenv(secret_id)
    if val:
        return val
    
    # 2. GCP Secret Manager
    if PROJECT_ID:
        try:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            print(f"Warning: Could not fetch secret {secret_id} from GCP: {e}")
    
    return default

GOOGLE_CLIENT_ID = get_secret("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = get_secret("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
GCS_IMAGES_BASE_URL = os.getenv("GCS_IMAGES_BASE_URL", "https://storage.googleapis.com/your-bucket-name/images")

# Admin Configuration
# Expects comma-separated emails: "admin@example.com,dev@example.com"
admin_emails_str = os.getenv("ADMIN_EMAILS", "")
ADMIN_EMAILS = [email.strip() for email in admin_emails_str.split(",") if email.strip()]
print(f"DEBUG: Configured Admin Emails: {ADMIN_EMAILS}")

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
        
        /* Chat header with book thumbnail */
        .chat-header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .chat-header-thumbnail {
            width: 80px;
            height: 120px;
            border-radius: 8px;
            border: 2px solid #00d9ff;
            box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
            object-fit: cover;
        }
        
        .chat-header-info {
            text-align: left;
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
        
        /* Hide Streamlit's default empty chat container */
        .stChatMessage {
            display: none !important;
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
    """Get book cover image URL from GCS"""
    # Return GCS URL directly
    return f"{GCS_IMAGES_BASE_URL}/{book_id}.png"

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

def clean_author_name(author):
    """Clean author name by removing GCS suffix"""
    if not author:
        return "Unknown Author"
    # Remove "(GCS)" or " (GCS)" from author names
    author = author.replace("(GCS)", "").replace(" (GCS)", "").strip()
    if not author or author.lower() == "unknown":
        return "Unknown Author"
    return author

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
                                # Log Role Status
                                if st.session_state.user_email in ADMIN_EMAILS:
                                    print(f"LOGIN: User {st.session_state.user_email} identified as ADMIN.")
                                    st.success(f"Welcome, {st.session_state.user_name}! (Admin Access Granted 🛡️)")
                                    time.sleep(2)
                                else:
                                    print(f"LOGIN: User {st.session_state.user_email} is a STANDARD user.")
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
                                
                            st.warning("Login link expired. Please click 'Sign in' again.")
                        else:
                            st.error(f"Login failed: {e}")
                        
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
    # Only show welcome message if NOT in chat
    if st.session_state.current_page != "Chat":
        st.markdown(f"<p style='color: #00d9ff; font-weight: 700; font-size: 16px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 20px;'>Welcome, {st.session_state.user_name}</p>", unsafe_allow_html=True)
    badge = ""
    if st.session_state.user_email in ADMIN_EMAILS:
        badge = " <span style='background: #00d9ff; color: #0a0e27; padding: 2px 6px; border-radius: 4px; font-size: 10px; vertical-align: middle;'>ADMIN</span>"
        
    st.markdown(f"<p style='color: #00d9ff; font-weight: 700; font-size: 16px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 20px;'>Welcome, {st.session_state.user_name}{badge}</p>", unsafe_allow_html=True)
    
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
    
    nav_options = ["Library", "My Activity", "Add New Book", "Health Check"]
    
    # Add Admin Dashboard
    if st.session_state.user_email and st.session_state.user_email in ADMIN_EMAILS:
        nav_options.append("Admin Dashboard")
        
    if st.session_state.selected_book:
        # Prepend Chat if a book is selected
        nav_options.insert(0, "Chat")
        
    page = st.radio("Go to", nav_options)

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

    # Chat header with book thumbnail
    title = book.get('title', 'Untitled')
    author = clean_author_name(book.get('author', ''))
    
    # Create header with thumbnail
    st.markdown('<div class="chat-header-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        try:
            st.markdown(f'<img src="{get_book_image_url(book["book_id"])}" class="chat-header-thumbnail" />', unsafe_allow_html=True)
        except:
            st.markdown('<div style="width: 80px; height: 120px; background: #0a0a0a; border: 2px dashed #00d9ff; border-radius: 8px; display: flex; align-items: center; justify-content: center;"><p style="color: #00d9ff; font-size: 10px; text-align: center;">No Image</p></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<h2 style='margin-bottom: 5px;'>{title}</h2>", unsafe_allow_html=True)
        if author != "Unknown Author":
            st.markdown(f"<p style='color: #a8b0c1; font-size: 16px; margin: 0;'>{author}</p>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()
    
    # Chat display area - only show if there are messages
    if st.session_state.messages:
        # Container for chat messages
        chat_container = st.container()
        with chat_container:
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
    else:
        # Compact empty state
        st.markdown("<div class='chat-empty-compact'><p style='color: #a8b0c1; font-size: 16px; margin: 0;'>Start asking questions about this book</p></div>", unsafe_allow_html=True)
    
    st.divider()
    
    # Chat input
    prompt = st.chat_input("Ask a question about this audiobook...", key="chat_input")
    
    if prompt:
        # Add user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Rerun to show the user message before processing
        st.rerun()

# Process the last message if it's a user message without a response
if page == "Chat" and st.session_state.selected_book and st.session_state.messages:
    if st.session_state.messages[-1]["role"] == "user":
        book = st.session_state.selected_book
        until_chapter = st.session_state.until_chapter
        until_time_total = st.session_state.until_time_total
        
        with st.spinner("Thinking..."):
            try:
                payload = {
                    "query": st.session_state.messages[-1]["content"],
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
                    error_msg = f"Error: {response.status_code} - {response.text}"
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
                
                # Clean author name
                clean_author = clean_author_name(book.get("author", ""))
                
                st.markdown(f'<div class="book-info"><p class="book-title">{book.get("title", "Untitled")}</p><p class="book-author">{clean_author}</p></div>', unsafe_allow_html=True)
                
                # Force refresh by using unique key without time (causes issues)
                button_key = f"chat_btn_{book['book_id']}_{i}"
                if st.button("Start Chat", key=button_key, use_container_width=True):
                    # Set all state variables
                    st.session_state.selected_book = book
                    st.session_state.messages = []
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.until_chapter = 0
                    st.session_state.until_time_total = 0
                    st.session_state.current_page = "Chat"
                    # Force immediate rerun
                    st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)

elif page == "Add New Book":
    st.header("Request New Book")
    tab_upload, tab_gcs = st.tabs(["Upload File", "Import from GCS"])
    
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
                                    
                                    # Check for email confirmation
                                    if job_info.get('email_sent'):
                                        st.success(f"Confirmation email sent to {st.session_state.user_email}")
                                    
                                    st.info("You'll receive an email when processing is complete!")
                                else:
                                    st.error(f"Submission failed: {process_response.status_code}")
                                    
                            except Exception as e:
                                st.error(f"Processing connection error: {e}")
                        else:
                            st.error(f"Upload failed: {response.status_code}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

    with tab_gcs:
        st.write("Import files already uploaded to Google Cloud Storage.")
        st.info("💡 Use this for large files (>32MB) if you are using Cloud Run.")
        
        with st.form("gcs_import_form"):
            gcs_book_name = st.text_input("Book Name (GCS)")
            gcs_path = st.text_input("GCS Folder Path (e.g., gs://my-bucket/audiobooks/harry_potter/)")
            gcs_submitted = st.form_submit_button("Start Processing")
            
            if gcs_submitted:
                if not gcs_book_name:
                    st.error("Please enter a book name.")
                elif not gcs_path:
                    st.error("Please enter the GCS path.")
                elif not gcs_path.startswith("gs://"):
                    st.error("Path must start with gs://")
                else:
                    processed_book_name = gcs_book_name.strip().lower().replace(" ", "_")
                    
                    with st.spinner("Submitting Job..."):
                        try:
                            process_payload = {
                                "folder_path": gcs_path,
                                "book_name": processed_book_name,
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
                                st.info("Check 'My Activity' for progress.")
                            else:
                                st.error(f"Submission failed: {process_response.status_code} - {process_response.text}")
                                
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
            elif response.status_code == 500:
                st.warning("⚠️ Upload history service is temporarily unavailable. Our team has been notified. Please try again later.")
            else:
                st.warning(f"Unable to load upload history at this time (Error {response.status_code}). Please try refreshing.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to the server. Please check your internet connection and try again.")
        except Exception as e:
            st.warning(f"Unable to load upload history: Service temporarily unavailable")
    
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
                st.info("💬 Chat history feature is coming soon! For now, your conversations are private and not stored.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to the server. Please check your internet connection and try again.")
        except Exception as e:
            st.info("💬 Chat history feature is coming soon! For now, your conversations are private and not stored.")

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

elif page == "Admin Dashboard":
    # Security Check
    if st.session_state.user_email not in ADMIN_EMAILS:
        print(f"SECURITY ALERT: Unauthorized admin access attempt by {st.session_state.user_email}")
        st.error("⛔ Access Denied: You are not authorized to view this page.")
        st.stop()
        
    print(f"ADMIN ACCESS: Granted to {st.session_state.user_email}")
    st.header("Admin Dashboard")
    
    # Tabs for organization
    tab_stats, tab_mlflow = st.tabs(["📊 System Metrics", "🧪 MLflow Experiments"])
    
    with tab_stats:
        st.info("System Statistics & Health")
        
        with st.expander("🛡️ Authorized Administrators"):
            for email in ADMIN_EMAILS:
                st.markdown(f"- `{email}`")
        
        if st.button("Refresh Stats"):
            st.rerun()

        known_book_ids = []
        try:
            with st.spinner("Fetching system statistics..."):
                response = requests.get(f"{API_URL}/admin/stats")
                
            if response.status_code == 200:
                stats = response.json()
                db_stats = stats.get("database", {})
                job_stats = stats.get("jobs", {})
                books = stats.get("books", [])
                
                # Populate for dropdown
                known_book_ids = [b['book_id'] for b in books]
                
                col_actions1, col_actions2 = st.columns([1, 4])
                with col_actions1:
                     # Button to trigger manual GCS sync
                    if st.button("☁️ Sync from Cloud"):
                        with st.spinner("Syncing metadata from GCS..."):
                            try:
                                sync_resp = requests.post(f"{API_URL}/admin/sync")
                                if sync_resp.status_code == 200:
                                    st.toast("Sync complete! Cloud books are now visible.", icon="✅")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error(f"Sync failed: {sync_resp.text}")
                            except Exception as e:
                                st.error(f"Sync connection failed: {e}")

                with col_actions2:
                     pass # Spacer
                
                # --- ROW 1: System Health & Job Stats ---
                # Initialize filter state
                if "job_filter" not in st.session_state:
                    st.session_state.job_filter = "Processing"

                # --- ROW 1: System Health & Job Stats ---
                st.subheader("System Overview")
                
                # Custom CSS to make buttons look like metrics cards
                st.markdown("""
                <style>
                div.stButton > button {
                    width: 100%;
                    height: 80px;
                    border-radius: 10px;
                    border: 1px solid #333;
                    background-color: #0e1117;
                    color: white;
                    font-size: 16px;
                }
                div.stButton > button:hover {
                    border-color: #00d9ff;
                    color: #00d9ff;
                }
                </style>
                """, unsafe_allow_html=True)

                col1, col2, col3, col4 = st.columns(4)
                
                # Turn Metrics into Buttons
                if col1.button(f"Total Books\n{db_stats.get('total_books', 0)}", key="btn_total_books"):
                    st.session_state.job_filter = "Books"
                
                if col2.button(f"Active Jobs\n{job_stats.get('processing', 0)}", key="btn_processing"):
                    st.session_state.job_filter = "Processing"
                    
                if col3.button(f"Completed Jobs\n{job_stats.get('completed', 0)}", key="btn_completed"):
                    st.session_state.job_filter = "Completed"
                    
                if col4.button(f"Failed Jobs\n{job_stats.get('failed', 0)}", key="btn_failed"):
                    st.session_state.job_filter = "Failed"
                
                # Current Filter
                job_status_filter = st.session_state.job_filter

                # === DISPLAY LOCIC ===
                
                if job_status_filter == "Books":
                    # Show Book Processing Status Table
                    st.subheader("Book Processing Status")
                    if books:
                        df = pd.DataFrame(books)
                        # Add a 'Status' column based on chunk count
                        df['Status'] = df['chunk_count'].apply(lambda x: '✅ Ready' if x > 0 else '⚠️ Empty/Processing')
                        
                        st.dataframe(
                            df,
                            column_config={
                                "book_id": "Book ID",
                                "title": "Title",
                                "chapter_count": "Chapters",
                                "chunk_count": "Chunks",
                                "Status": "Status"
                            },
                            use_container_width=True,
                            hide_index=True
                        )
                    else:
                        st.info("No books found in the system.")
                        
                else:
                    # Show Job Status Table
                    status_key = job_status_filter.lower()
                    
                    # Fetch filtered jobs
                    try:
                        jobs_resp = requests.get(f"{API_URL}/admin/jobs", params={"status": status_key})
                        if jobs_resp.status_code == 200:
                            filtered_jobs = jobs_resp.json().get("jobs", [])
                        else:
                            st.error(f"Failed to fetch {job_status_filter} jobs")
                            filtered_jobs = []
                    except Exception as e:
                        st.error(f"Connection error: {e}")
                        filtered_jobs = []

                    with st.expander(f"{job_status_filter} Jobs ({len(filtered_jobs)})", expanded=True):
                        if filtered_jobs:
                            f_df = pd.DataFrame(filtered_jobs)
                            
                            # Convert created_at to Local Time (US/Eastern)
                            if "created_at" in f_df.columns:
                                f_df["created_at"] = pd.to_datetime(f_df["created_at"])
                                # Ensure it's timezone aware (localize as UTC if naive)
                                if f_df["created_at"].dt.tz is None:
                                    f_df["created_at"] = f_df["created_at"].dt.tz_localize("UTC")
                                # Convert to Eastern Time
                                f_df["created_at"] = f_df["created_at"].dt.tz_convert("US/Eastern")

                            cols_to_show = ["job_id", "book_name", "status", "progress", "message", "created_at"]
                            # Handle potential missing columns
                            display_cols = [c for c in cols_to_show if c in f_df.columns]
                            f_display = f_df[display_cols]
                            
                            st.dataframe(
                                f_display, 
                                column_config={
                                    "progress": st.column_config.ProgressColumn(
                                        "Progress",
                                        format="%.2f",
                                        min_value=0,
                                        max_value=1,
                                    ),
                                    "created_at": st.column_config.DatetimeColumn(
                                        "Created At (EST)",
                                        format="D MMM, HH:mm"
                                    )
                                },
                                use_container_width=True,
                                hide_index=True
                            )
                        else:
                            st.info(f"No jobs found with status '{job_status_filter}'.")


                # st.json(stats) # Hidden for production UI
                
            else:
                st.error(f"Failed to fetch stats: {response.status_code}")
        except Exception as e:
            err_msg = str(e)
            if "Max retries exceeded" in err_msg or "Connection refused" in err_msg:
                st.warning("Backend starting... Auto-refreshing in 3s ⏳")
                time.sleep(3)
                st.rerun()
            else:
                st.error(f"Connection error: {e}")


                    
    with tab_mlflow:
        st.subheader("🧪 MLflow Experiments (Pro Dashboard)")
        st.caption("Live performance tracking from the MLflow Server")
        
        try:
            import mlflow
            from mlflow.entities import ViewType
            import pandas as pd
            
            mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            
            # Embed the MLflow UI
            st.markdown(f"### 🖥️ MLflow Dashboard")
            st.info(f"The MLflow dashboard is hosted separately at `{mlflow_tracking_uri}`.")
            st.link_button("↗️ Open MLflow Dashboard", mlflow_tracking_uri)
            
            st.divider()
            
            st.divider()


            
        except Exception as e:
            st.error(f"Failed to connect to MLflow: {e}")
            st.warning("Ensure the 'mlflow' service is running.")
            st.caption("Common fix: Make sure the MLflow Deployment is running in the K8s cluster.")
