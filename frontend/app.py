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

st.set_page_config(page_title="AudioSeek", layout="wide", initial_sidebar_state="expanded")

# ========================================================================
# ULTRA-PREMIUM DARK THEME CSS
# ========================================================================
st.markdown("""
    <style>
        /* Global Background */
        html, body, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
            background: linear-gradient(135deg, #0a0e27 0%, #1a0f3a 50%, #0f1a2e 100%) !important;
        }
        
        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f1629 0%, #1a1f3a 100%) !important;
            border-right: 3px solid #00d9ff;
        }
        
        [data-testid="stSidebarContent"] {
            background: transparent !important;
        }
        
        /* Main Content */
        .stMainBlockContainer {
            background: transparent !important;
        }
        
        /* Typography */
        h1 {
            background: linear-gradient(135deg, #00d9ff 0%, #0099cc 50%, #00ffff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
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
        
        p, span, label {
            color: #e4e6eb !important;
        }
        
        /* Book Cards - Premium Design */
        .book-card {
            position: relative;
            background: linear-gradient(135deg, #16213e 0%, #0f1629 100%);
            border: 2px solid #00d9ff;
            border-radius: 16px;
            padding: 20px;
            transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
            box-shadow: 
                0 0 30px rgba(0, 217, 255, 0.2),
                inset 0 1px 0 rgba(255, 255, 255, 0.1);
            overflow: hidden;
        }
        
        .book-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(135deg, rgba(0, 217, 255, 0.1) 0%, transparent 100%);
            opacity: 0;
            transition: opacity 0.4s ease;
            pointer-events: none;
        }
        
        .book-card:hover {
            border-color: #00ffff;
            transform: translateY(-12px) scale(1.02);
            box-shadow: 
                0 20px 60px rgba(0, 217, 255, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.2),
                0 0 40px rgba(0, 255, 255, 0.3);
        }
        
        .book-card:hover::before {
            opacity: 1;
        }
        
        .book-cover-image {
            width: 100%;
            height: 320px;
            border-radius: 12px;
            object-fit: cover;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.5);
            transition: all 0.4s ease;
            border: 1px solid rgba(0, 217, 255, 0.3);
        }
        
        .book-card:hover .book-cover-image {
            box-shadow: 0 12px 40px rgba(0, 217, 255, 0.3);
            border-color: #00d9ff;
        }
        
        .book-cover-placeholder {
            width: 100%;
            height: 320px;
            background: linear-gradient(135deg, #1a2a4a 0%, #0a1a3a 100%);
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
            box-shadow: inset 0 0 20px rgba(0, 217, 255, 0.1);
        }
        
        /* Small Book Thumbnail in Chat */
        .chat-book-thumbnail {
            width: 120px;
            height: 180px;
            border-radius: 8px;
            object-fit: cover;
            box-shadow: 0 4px 12px rgba(0, 217, 255, 0.3);
            border: 2px solid #00d9ff;
        }
        
        .book-info {
            margin: 16px 0;
            position: relative;
            z-index: 1;
        }
        
        .book-title {
            font-size: 18px;
            font-weight: 700;
            color: #ffffff;
            margin: 12px 0 8px 0;
            line-height: 1.3;
        }
        
        .book-author {
            font-size: 13px;
            color: #a8b0c1;
            margin: 0;
            font-style: italic;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%) !important;
            color: #0a0e27 !important;
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
            box-shadow: 0 8px 25px rgba(0, 217, 255, 0.5) !important;
            background: linear-gradient(135deg, #00ffff 0%, #00d9ff 100%) !important;
        }
        
        /* Input Fields */
        .stTextInput > div > div > input {
            background-color: rgba(26, 31, 58, 0.8) !important;
            border: 2px solid #2a3550 !important;
            color: #e4e6eb !important;
            border-radius: 10px !important;
            padding: 12px 16px !important;
            font-size: 14px !important;
            transition: all 0.3s ease !important;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #00d9ff !important;
            box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.2) !important;
            background-color: rgba(26, 31, 58, 0.95) !important;
        }
        
        /* Chat Input */
        .stChatInputContainer {
            background: linear-gradient(135deg, rgba(26, 31, 58, 0.5) 0%, rgba(15, 22, 41, 0.5) 100%) !important;
            border-top: 2px solid #00d9ff;
            padding: 20px !important;
            border-radius: 0 !important;
        }
        
        /* Chat Messages */
        .chat-message-user {
            background: linear-gradient(135deg, #00d9ff 0%, #00b8cc 100%);
            color: #0a0e27;
            border-radius: 16px;
            padding: 16px 20px;
            margin: 12px 0;
            font-weight: 600;
            box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
            max-width: 75%;
            margin-left: auto;
            word-wrap: break-word;
            font-size: 15px;
            line-height: 1.5;
        }
        
        .chat-message-assistant {
            background: linear-gradient(135deg, #1a2a4a 0%, #0f1a3a 100%);
            color: #e4e6eb;
            border: 2px solid #00d9ff;
            border-radius: 16px;
            padding: 16px 20px;
            margin: 12px 0;
            box-shadow: 0 4px 15px rgba(0, 217, 255, 0.2);
            max-width: 85%;
            word-wrap: break-word;
            font-size: 15px;
            line-height: 1.6;
        }
        
        /* Sidebar Navigation */
        [data-testid="stRadio"] label {
            color: #e4e6eb !important;
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
            opacity: 0.3;
        }
        
        /* Status Messages */
        .stSuccess {
            background-color: rgba(26, 42, 26, 0.8) !important;
            color: #4ade80 !important;
            border-left: 4px solid #4ade80 !important;
            border-radius: 8px !important;
        }
        
        .stError {
            background-color: rgba(42, 26, 26, 0.8) !important;
            color: #f87171 !important;
            border-left: 4px solid #f87171 !important;
            border-radius: 8px !important;
        }
        
        .stInfo {
            background-color: rgba(26, 31, 58, 0.8) !important;
            color: #00d9ff !important;
            border-left: 4px solid #00d9ff !important;
            border-radius: 8px !important;
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

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "previous_chapter" not in st.session_state:
    st.session_state.previous_chapter = 0

if "previous_time" not in st.session_state:
    st.session_state.previous_time = 0

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================
def fetch_books():
    """Fetch books from backend API with fallback to default list"""
    default_books = [
        {"book_id": "around_the_world", "title": "Around the World in 80 Days", "author": "Jules Verne"},
        {"book_id": "harry_potter_and_philosopher_stone", "title": "Harry Potter and the Philosopher's Stone", "author": "J.K. Rowling"},
        {"book_id": "harry_potter_and_sorcerer_s_stone", "title": "Harry Potter and the Sorcerer's Stone", "author": "J.K. Rowling"},
        {"book_id": "romeo_and_juliet", "title": "Romeo and Juliet", "author": "William Shakespeare"}
    ]
    
    try:
        response = requests.get(f"{API_URL}/books", timeout=5)
        if response.status_code == 200:
            backend_books = response.json()
            if backend_books:
                return backend_books
    except:
        pass
    
    return default_books

def get_book_image_url(book_id):
    """Get GCS URL or local path for book cover image"""
    # Try GCS first (you'll need to implement this endpoint)
    try:
        response = requests.get(f"{API_URL}/books/{book_id}/cover", timeout=3)
        if response.status_code == 200:
            data = response.json()
            if "cover_url" in data:
                return data["cover_url"]
    except:
        pass
    
    # Fallback to local images
    return f"./images/{book_id}.png"

# ========================================================================
# AUTHENTICATION - GOOGLE OAUTH
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
                
                auth_url, state = flow.authorization_url()
                
                st.markdown(f"""
                    <div style='text-align: center; margin-top: 40px;'>
                        <a href='{auth_url}' style='
                            display: inline-block;
                            background: linear-gradient(135deg, #00d9ff 0%, #0099cc 100%);
                            color: #0a0e27;
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
                
                query_params = st.query_params
                if "code" in query_params:
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
                                st.session_state.authenticated = True
                                st.success(f"Welcome, {st.session_state.user_name}!")
                                time.sleep(1)
                                st.rerun()
                    except Exception as e:
                        st.error(f"Token exchange failed: {e}")
                        
            except Exception as e:
                st.error(f"Authentication error: {e}")
        else:
            st.warning("Google OAuth credentials not configured in .env.local")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.stop()

# ========================================================================
# MAIN APPLICATION (Only reachable if logged in)
# ========================================================================

# Sidebar
with st.sidebar:
    st.markdown(f"<p style='color: #00d9ff; font-weight: 700; font-size: 16px; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 20px;'>Welcome, {st.session_state.user_name}</p>", unsafe_allow_html=True)
    
    if st.button("Sign Out", use_container_width=True):
        st.session_state.user_email = None
        st.session_state.user_name = None
        st.session_state.authenticated = False
        st.session_state.selected_book = None
        st.session_state.messages = []
        st.query_params.clear()
        st.rerun()
    
    st.divider()
    st.header("Navigation")
    
    if st.session_state.selected_book:
        page = st.radio("Go to", ["Chat", "Library", "My Activity", "Add New Book", "Health Check"])
    else:
        page = st.radio("Go to", ["Library", "My Activity", "Add New Book", "Health Check"])

# ========================================================================
# PAGE: CHAT (Dedicated full-screen chat)
# ========================================================================
if page == "Chat" and st.session_state.selected_book:
    book = st.session_state.selected_book
    
    # Sidebar adjustments for Chat
    with st.sidebar:
        st.divider()
        st.subheader("Spoiler Control")
        st.caption("Restrict answers to progress (0 = search all)")
        
        until_chapter = st.number_input(
            "Until Chapter", 
            min_value=0, 
            value=st.session_state.previous_chapter, 
            help="0 = searching all chapters",
            key="chapter_input"
        )
        
        c1, c2 = st.columns(2)
        with c1:
            until_minutes = st.number_input(
                "Minutes", 
                min_value=0, 
                value=st.session_state.previous_time // 60, 
                step=1,
                key="minutes_input"
            )
        with c2:
            until_seconds = st.number_input(
                "Seconds", 
                min_value=0, 
                value=st.session_state.previous_time % 60, 
                step=1,
                key="seconds_input"
            )
            
        until_time_total = (until_minutes * 60) + until_seconds
        
        # Check if progress changed
        if (until_chapter != st.session_state.previous_chapter or 
            until_time_total != st.session_state.previous_time):
            
            if st.button("⚠️ Confirm Progress Change", use_container_width=True, type="primary"):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.session_state.previous_chapter = until_chapter
                st.session_state.previous_time = until_time_total
                st.success("Session reset! Progress updated.")
                time.sleep(1)
                st.rerun()

    # Header with book info and thumbnail
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        # Book thumbnail
        image_url = get_book_image_url(book['book_id'])
        try:
            st.image(image_url, use_column_width=True)
        except:
            st.markdown('<div class="book-cover-placeholder" style="height: 150px; width: 100px;">Cover</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"<h2 style='text-align: center; margin-bottom: 5px;'>{book.get('title')}</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: #a8b0c1; font-size: 16px; margin: 0;'>{book.get('author')}</p>", unsafe_allow_html=True)
    
    st.divider()
    
    # Chat messages container
    if st.session_state.messages:
        # Use a scrollable container with dynamic height
        messages_html = "<div style='max-height: 500px; overflow-y: auto; padding: 24px; background: linear-gradient(135deg, rgba(26, 31, 58, 0.4) 0%, rgba(15, 22, 41, 0.4) 100%); border-radius: 12px; border: 2px solid #00d9ff;'>"
        
        for message in st.session_state.messages:
            if message["role"] == "user":
                messages_html += f'<div class="chat-message-user">{message["content"]}</div>'
            else:
                messages_html += f'<div class="chat-message-assistant">{message["content"]}</div>'
        
        messages_html += "</div>"
        st.markdown(messages_html, unsafe_allow_html=True)
        
        # Display audio references separately
        for message in st.session_state.messages:
            if message["role"] == "assistant" and "audio" in message:
                for ref in message["audio"]:
                    if "url" in ref:
                        start_time = int(ref.get("start_time", 0))
                        st.audio(ref["url"], start_time=start_time)
                        st.caption(f"Chapter {ref.get('chapter_id')} at {start_time}s")
    else:
        st.info("Start asking questions about this book...")
    
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
    
    # Fetch books once and cache in session state
    if "cached_books" not in st.session_state:
        st.session_state.cached_books = fetch_books()
    
    books = st.session_state.cached_books
    
    if st.button("🔄 Refresh Library"):
        st.session_state.cached_books = fetch_books()
        st.rerun()

    if not books:
        st.info("No books found in the library.")
    else:
        cols = st.columns(2, gap="large")
        for i, book in enumerate(books):
            with cols[i % 2]:
                st.markdown('<div class="book-card">', unsafe_allow_html=True)
                
                image_url = get_book_image_url(book['book_id'])
                try:
                    st.image(image_url, use_column_width=True)
                except:
                    st.markdown('<div class="book-cover-placeholder">Book Cover</div>', unsafe_allow_html=True)
                
                st.markdown(f'<div class="book-info"><p class="book-title">{book.get("title", "Unknown")}</p><p class="book-author">{book.get("author", "Unknown")}</p></div>', unsafe_allow_html=True)
                
                if st.button("Chat", key=f"chat_{book['book_id']}_{i}", use_container_width=True):
                    st.session_state.selected_book = book
                    st.session_state.messages = []
                    st.session_state.session_id = str(uuid.uuid4())
                    st.session_state.previous_chapter = 0
                    st.session_state.previous_time = 0
                    st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)

# ========================================================================
# PAGE: ADD NEW BOOK
# ========================================================================
elif page == "Add New Book":
    st.header("Request New Book")
    st.write("Upload an audio file (MP3/WAV/M4A/OGG/FLAC) or a ZIP file containing audio chapters.")
    
    with st.form("upload_form"):
        book_name = st.text_input("Book Name")
        uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "m4a", "ogg", "flac", "zip"])
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
                                    
                                    # Send email notification
                                    try:
                                        email_payload = {
                                            "to_email": st.session_state.user_email,
                                            "subject": f"AudioSeek: Processing started for '{book_name}'",
                                            "body": f"Your audiobook '{book_name}' has been uploaded and processing has started. Job ID: {job_info.get('job_id')}"
                                        }
                                        requests.post(f"{API_URL}/send-email", json=email_payload)
                                    except:
                                        pass
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
        
        if st.button("Refresh Upload Status"):
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
                                st.markdown(f"<p style='font-size: 18px; font-weight: 700; color: #ffffff; margin: 0;'>{job.get('book_name')}</p>", unsafe_allow_html=True)
                                st.markdown(f"<p style='font-size: 13px; color: #a8b0c1; margin: 5px 0 0 0;'>Job ID: <code>{job.get('job_id')}</code></p>", unsafe_allow_html=True)
                            
                            with col2:
                                status = job.get('status', 'unknown')
                                if status == 'processing':
                                    st.markdown("<span style='background: #1a2a1a; color: #4ade80; padding: 6px 12px; border-radius: 6px; font-weight: 600; font-size: 12px;'>PROCESSING</span>", unsafe_allow_html=True)
                                elif status == 'completed':
                                    st.markdown("<span style='background: #1a2a1a; color: #4ade80; padding: 6px 12px; border-radius: 6px; font-weight: 600; font-size: 12px;'>COMPLETED</span>", unsafe_allow_html=True)
                                elif status == 'failed':
                                    st.markdown("<span style='background: #2a1a1a; color: #f87171; padding: 6px 12px; border-radius: 6px; font-weight: 600; font-size: 12px;'>FAILED</span>", unsafe_allow_html=True)
                            
                            with col3:
                                if status == 'processing':
                                    st.progress(job.get('progress', 0.0))
                            
                            if job.get('message'):
                                st.markdown(f"<p style='color: #a8b0c1; font-size: 12px; margin: 8px 0 0 0;'>{job.get('message')}</p>", unsafe_allow_html=True)
            else:
                st.error(f"Failed to fetch upload history: {response.status_code}")
        except Exception as e:
            st.info(f"Upload history not available: {e}")
    
    with tab2:
        st.subheader("Your Chat History")
        
        try:
            response = requests.get(f"{API_URL}/chat-history/{st.session_state.user_email}")
            if response.status_code == 200:
                chat_history = response.json()
                if not chat_history:
                    st.info("No chat history yet. Start a conversation with a book!")
                else:
                    for chat in chat_history:
                        with st.container(border=True):
                            st.markdown(f"<p style='font-size: 16px; font-weight: 700; color: #00d9ff;'>{chat.get('book_title', 'Unknown Book')}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #a8b0c1; font-size: 12px; margin: 5px 0;'>{chat.get('timestamp', 'N/A')}</p>", unsafe_allow_html=True)
                            
                            st.markdown(f"<p style='color: #e4e6eb; margin: 10px 0; font-style: italic;'>You: {chat.get('question', '')}</p>", unsafe_allow_html=True)
                            st.markdown(f"<p style='color: #a8b0c1; margin: 0;'>AudioSeek: {chat.get('answer', '')[:200]}...</p>", unsafe_allow_html=True)
            else:
                st.info("Chat history not yet available from backend")
        except Exception as e:
            st.info("Chat history feature coming soon. Start chatting to build your history!")

# ========================================================================
# PAGE: HEALTH CHECK
# ========================================================================
elif page == "Health Check":
    st.header("Service Health")
    
    if st.button("Check Health"):
        try:
            response = requests.get(f"{API_URL}/health")
            st.json(response.json())
        except Exception as e:
            st.error(f"Failed to connect to service: {e}")