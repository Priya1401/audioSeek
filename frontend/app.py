import streamlit as st
import requests
import os
import uuid
from dotenv import load_dotenv
import base64
import json
import time
import pandas as pd
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests
from google.cloud import secretmanager
from google.cloud import storage
import datetime

# Load environment variables
load_dotenv(".env.local")

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8001")
PROJECT_ID = os.getenv("PROJECT_ID")

def get_secret(secret_id, default=None):
    val = os.getenv(secret_id)
    if val:
        return val
    
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

admin_emails_str = os.getenv("ADMIN_EMAILS", "")
ADMIN_EMAILS = [email.strip() for email in admin_emails_str.split(",") if email.strip()]

st.set_page_config(page_title="AudioSeek", layout="wide", initial_sidebar_state="expanded")

# ========================================================================
# PREMIUM DARK THEME CSS
# ========================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    
    /* Global Dark Background */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: linear-gradient(180deg, #0a0a0f 0%, #0d1117 50%, #0a0a0f 100%);
    }
    
    section[data-testid="stSidebar"] {
        background: rgba(10, 10, 15, 0.95);
        border-right: 1px solid rgba(0, 217, 255, 0.2);
    }
    
    /* Page Title - Custom Class */
    .page-title {
        color: #00d9ff !important;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        letter-spacing: -1px;
        margin-bottom: 10px;
        text-shadow: 0 0 30px rgba(0, 217, 255, 0.3);
    }
    
    .page-subtitle {
        color: #64748b;
        font-size: 14px;
        margin-bottom: 20px;
    }
    
    /* Default text colors */
    h1, h2, h3 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    p, span, label, div {
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Card Container Styling */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, rgba(15, 15, 25, 0.9) 0%, rgba(10, 10, 20, 0.95) 100%);
        border: 1px solid rgba(0, 217, 255, 0.2);
        border-radius: 20px;
        transition: all 0.4s ease;
    }
    
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(0, 217, 255, 0.5);
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4), 0 0 40px rgba(0, 217, 255, 0.1);
    }
    
    /* Book Cover Placeholder */
    .book-cover {
        width: 100%;
        height: 280px;
        background: linear-gradient(145deg, #1a1a2e 0%, #0f0f1a 100%);
        border-radius: 12px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 1px solid rgba(0, 217, 255, 0.15);
        margin-bottom: 16px;
        position: relative;
        overflow: hidden;
    }
    
    .book-cover::before {
        content: '';
        position: absolute;
        width: 100px;
        height: 100px;
        background: radial-gradient(circle, rgba(0, 217, 255, 0.1) 0%, transparent 70%);
        border-radius: 50%;
    }
    
    .book-cover-title {
        color: #00d9ff;
        font-size: 15px;
        font-weight: 600;
        text-align: center;
        padding: 20px;
        z-index: 1;
        line-height: 1.4;
    }
    
    .book-title-text {
        color: #ffffff !important;
        font-size: 17px;
        font-weight: 700;
        margin: 8px 0;
        line-height: 1.3;
    }
    
    .book-author-text {
        color: #94a3b8 !important;
        font-size: 14px;
        font-weight: 400;
        margin-bottom: 16px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d9ff 0%, #0891b2 100%);
        color: #000000 !important;
        border: none;
        border-radius: 12px;
        font-weight: 700;
        font-size: 14px;
        padding: 12px 24px;
        text-transform: uppercase;
        letter-spacing: 1px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 217, 255, 0.5);
        background: linear-gradient(135deg, #00ffff 0%, #00d9ff 100%);
    }
    
    /* Secondary Buttons */
    button[kind="secondary"] {
        background: transparent !important;
        border: 2px solid rgba(0, 217, 255, 0.5) !important;
        color: #00d9ff !important;
    }
    
    /* Input Fields */
    .stTextInput input, .stNumberInput input {
        background: rgba(15, 15, 25, 0.8) !important;
        border: 1px solid rgba(0, 217, 255, 0.2) !important;
        border-radius: 10px !important;
        color: #ffffff !important;
        padding: 12px 16px !important;
    }
    
    .stTextInput input:focus, .stNumberInput input:focus {
        border-color: #00d9ff !important;
        box-shadow: 0 0 0 2px rgba(0, 217, 255, 0.2) !important;
    }
    
    /* Chat Messages */
    .user-message {
        background: linear-gradient(135deg, #00d9ff 0%, #0891b2 100%);
        color: #000000 !important;
        padding: 16px 20px;
        border-radius: 18px 18px 4px 18px;
        margin: 12px 0;
        margin-left: 20%;
        font-weight: 500;
    }
    
    .assistant-message {
        background: rgba(20, 20, 35, 0.9);
        color: #e2e8f0 !important;
        padding: 16px 20px;
        border-radius: 18px 18px 18px 4px;
        margin: 12px 0;
        margin-right: 20%;
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    /* Empty Chat State */
    .empty-chat {
        text-align: center;
        padding: 60px 20px;
        background: rgba(15, 15, 25, 0.5);
        border-radius: 16px;
        border: 1px dashed rgba(0, 217, 255, 0.3);
        margin: 20px 0;
    }
    
    .empty-chat p {
        color: #64748b !important;
        font-size: 16px;
    }
    
    /* Welcome Badge */
    .welcome-badge {
        background: rgba(0, 217, 255, 0.1);
        border: 1px solid rgba(0, 217, 255, 0.2);
        border-radius: 10px;
        padding: 12px 16px;
        margin-bottom: 20px;
    }
    
    .welcome-badge p {
        color: #00d9ff !important;
        font-weight: 600;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 0;
    }
    
    .admin-tag {
        background: linear-gradient(135deg, #00d9ff, #8b5cf6);
        color: #000 !important;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 700;
        margin-left: 8px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        background: transparent;
        border-radius: 8px;
        padding: 10px 20px;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00d9ff !important;
        background: rgba(0, 217, 255, 0.1);
    }
    
    /* Divider */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(0, 217, 255, 0.3), transparent) !important;
        margin: 20px 0 !important;
    }
    
    /* Status Messages */
    .stSuccess {
        background: rgba(34, 197, 94, 0.1) !important;
        border: 1px solid rgba(34, 197, 94, 0.3) !important;
        border-radius: 10px !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 10px !important;
    }
    
    .stInfo {
        background: rgba(0, 217, 255, 0.1) !important;
        border: 1px solid rgba(0, 217, 255, 0.3) !important;
        border-radius: 10px !important;
    }
    
    .stWarning {
        background: rgba(251, 191, 36, 0.1) !important;
        border: 1px solid rgba(251, 191, 36, 0.3) !important;
        border-radius: 10px !important;
    }
    
    /* Form Container */
    [data-testid="stForm"] {
        background: rgba(15, 15, 25, 0.6) !important;
        border: 1px solid rgba(0, 217, 255, 0.15) !important;
        border-radius: 16px !important;
        padding: 24px !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00d9ff, #8b5cf6) !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00d9ff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    
    /* Login Page */
    .login-box {
        text-align: center;
        padding: 80px 20px;
    }
    
    .login-title {
        font-size: 4rem;
        font-weight: 900;
        color: #00d9ff;
        margin-bottom: 10px;
        letter-spacing: -2px;
        text-shadow: 0 0 40px rgba(0, 217, 255, 0.4);
    }
    
    .login-subtitle {
        color: #64748b;
        font-size: 14px;
        letter-spacing: 3px;
        text-transform: uppercase;
        margin-bottom: 50px;
    }
    
    .login-btn {
        display: inline-block;
        background: linear-gradient(135deg, #00d9ff 0%, #8b5cf6 100%);
        color: #000000 !important;
        padding: 16px 48px;
        border-radius: 14px;
        text-decoration: none;
        font-weight: 700;
        font-size: 15px;
        letter-spacing: 2px;
        text-transform: uppercase;
        box-shadow: 0 8px 30px rgba(0, 217, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .login-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(0, 217, 255, 0.5);
        color: #000000 !important;
    }
    
    /* Subheader styling */
    .stSubheader {
        color: #ffffff !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        color: #e2e8f0 !important;
    }
    
    .stRadio [data-baseweb="radio"] {
        background: transparent;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(15, 15, 25, 0.6) !important;
        border: 1px solid rgba(0, 217, 255, 0.15) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
    }
    
    /* Caption */
    .stCaption {
        color: #64748b !important;
    }
    
    /* Code blocks */
    code {
        background: rgba(0, 217, 255, 0.1) !important;
        color: #00d9ff !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
    
    /* DataFrame */
    .stDataFrame {
        border: 1px solid rgba(0, 217, 255, 0.2) !important;
        border-radius: 10px !important;
    }
    
    /* Link buttons */
    .stLinkButton > a {
        background: linear-gradient(135deg, #00d9ff 0%, #0891b2 100%) !important;
        color: #000000 !important;
        border-radius: 10px !important;
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
if "current_page" not in st.session_state:
    st.session_state.current_page = "Library"

# ========================================================================
# AUTH PERSISTENCE FUNCTIONS
# ========================================================================
def save_auth_to_params():
    """Save auth state to query params for persistence across refreshes"""
    if st.session_state.authenticated and st.session_state.user_email:
        st.query_params["auth"] = "1"
        st.query_params["email"] = st.session_state.user_email
        st.query_params["name"] = st.session_state.user_name or "User"

def restore_auth_from_params():
    """Restore auth state from query params on page load"""
    params = st.query_params
    if params.get("auth") == "1" and params.get("email"):
        st.session_state.authenticated = True
        st.session_state.user_email = params.get("email")
        st.session_state.user_name = params.get("name", "User")
        return True
    return False

def clear_auth_params():
    """Clear auth params on logout"""
    for key in ["auth", "email", "name", "code"]:
        if key in st.query_params:
            del st.query_params[key]

# Try to restore auth on page load if not already authenticated
if not st.session_state.authenticated:
    restore_auth_from_params()

# ========================================================================
# HELPER FUNCTIONS
# ========================================================================
# Initialize Storage Client
try:
    storage_client = storage.Client()
    bucket = storage_client.bucket("audioseek-bucket")
except Exception as e:
    print(f"Failed to initialize storage client: {e}")
    storage_client = None

def format_dataframe_dates(df):
    """
    Format date columns in DataFrame to readable EST format.
    Target columns: created_at, updated_at, timestamp
    """
    date_cols = ['created_at', 'updated_at', 'timestamp']
    # Define EST offset (UTC-5)
    est_offset = pd.Timedelta(hours=-5)
    
    for col in date_cols:
        if col in df.columns:
            try:
                # Convert to datetime if not already
                df[col] = pd.to_datetime(df[col])
                
                # If timezone naive, assume UTC first
                if df[col].dt.tz is None:
                    df[col] = df[col].dt.tz_localize('UTC')
                
                # Convert to UTC then to fixed offset for EST
                # Using fixed offset avoid pytz dependency issues in some envs
                # But typically direct conversion is better. 
                # Let's simple format: Convert to UTC, add offset, remove tz
                
                # Convert to UTC
                df[col] = df[col].dt.tz_convert('UTC')
                
                # Shift to EST
                df[col] = df[col] + est_offset
                
                # Format to string without timezone info
                df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            except Exception as e:
                # If conversion fails, keep original
                pass
    return df

def get_book_image_url(book_id):
    if not storage_client:
        return None
        
    # Try extensions in order
    for ext in ["png", "jpg", "jpeg"]:
        blob_path = f"images/{book_id}.{ext}"
        blob = bucket.blob(blob_path)
        if blob.exists():
            return blob.generate_signed_url(
                version="v4",
                expiration=datetime.timedelta(minutes=15),
                method="GET"
            )
    return None

def fetch_books_from_api():
    try:
        response = requests.get(f"{API_URL}/books", timeout=5)
        if response.status_code == 200:
            books = response.json()
            return books if books else []
    except Exception as e:
        st.error(f"Failed to fetch books: {e}")
    return []

def clean_author_name(author):
    if not author:
        return "Unknown Author"
    author = author.replace("(GCS)", "").replace(" (GCS)", "").strip()
    if not author or author.lower() == "unknown":
        return "Unknown Author"
    return author

def select_book_for_chat(book):
    st.session_state.selected_book = book
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.until_chapter = 0
    st.session_state.until_time_total = 0
    st.session_state.current_page = "Chat"

def render_page_header(title, subtitle=None, show_refresh=False, refresh_key=None):
    """Render a consistent page header with optional refresh button"""
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
        if subtitle:
            st.markdown(f'<div class="page-subtitle">{subtitle}</div>', unsafe_allow_html=True)
    with col2:
        if show_refresh:
            st.write("")  # Spacing to align button
            if st.button("Refresh", key=refresh_key, use_container_width=True):
                st.rerun()
    st.divider()

# ========================================================================
# AUTHENTICATION
# ========================================================================
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="login-box">
                <div class="login-title">AudioSeek</div>
                <div class="login-subtitle">Discover Audiobook Insights Powered by AI</div>
            </div>
        """, unsafe_allow_html=True)
        
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
                    scopes=["https://www.googleapis.com/auth/userinfo.email", 
                            "https://www.googleapis.com/auth/userinfo.profile", "openid"]
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
                                
                                # Clear the OAuth code and save auth params
                                st.query_params.clear()
                                save_auth_to_params()
                                st.rerun()
                    except Exception as e:
                        st.error(f"Authentication failed: {e}")
                        # Clear bad code
                        if "code" in st.query_params:
                            del st.query_params["code"]
                else:
                    auth_url, state = flow.authorization_url()
                    st.markdown(f"""
                        <div style="text-align: center;">
                            <a href="{auth_url}" class="login-btn">Sign in with Google</a>
                        </div>
                    """, unsafe_allow_html=True)
                        
            except Exception as e:
                st.error(f"Authentication error: {e}")
        else:
            st.warning("Google OAuth credentials not configured")
    
    st.stop()

# Save auth state to params after successful authentication (keeps session alive on refresh)
save_auth_to_params()

# ========================================================================
# SIDEBAR
# ========================================================================
with st.sidebar:
    # Welcome message
    if st.session_state.current_page != "Chat":
        admin_tag = ""
        if st.session_state.user_email in ADMIN_EMAILS:
            admin_tag = "<span class='admin-tag'>ADMIN</span>"
        
        st.markdown(f"""
            <div class="welcome-badge">
                <p>Welcome, {st.session_state.user_name}{admin_tag}</p>
            </div>
        """, unsafe_allow_html=True)
    
    if st.button("Sign Out", use_container_width=True):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        # Clear auth params from URL
        clear_auth_params()
        st.rerun()
    
    st.divider()
    
    # Back button when in Chat
    if st.session_state.current_page == "Chat" and st.session_state.selected_book:
        if st.button("Back to Library", use_container_width=True, type="secondary"):
            st.session_state.current_page = "Library"
            st.session_state.selected_book = None
            st.session_state.messages = []
            st.rerun()
        st.divider()
        
        # Show current book
        st.subheader("Currently Reading")
        book = st.session_state.selected_book
        st.markdown(f"""
            <div class="book-cover" style="height: 180px;">
                <div class="book-cover-title">{book.get('title', 'Book')}</div>
            </div>
        """, unsafe_allow_html=True)
        st.divider()
    
    # Navigation
    st.subheader("Navigation")
    
    nav_options = ["Library", "My Activity", "Add New Book", "Health Check"]
    
    if st.session_state.user_email and st.session_state.user_email in ADMIN_EMAILS:
        nav_options.append("Admin Dashboard")
        
    if st.session_state.selected_book:
        nav_options.insert(0, "Chat")
        
    page = st.radio("Go to", nav_options, label_visibility="collapsed")
    
    if page != "Chat":
        st.session_state.current_page = page

# ========================================================================
# PAGE: CHAT
# ========================================================================
if page == "Chat" and st.session_state.selected_book:
    st.session_state.current_page = "Chat"
    book = st.session_state.selected_book
    
    # Spoiler controls in sidebar
    with st.sidebar:
        st.divider()
        st.subheader("Spoiler Control")
        st.caption("Restrict answers to your progress")
        
        current_chapter = st.session_state.until_chapter
        current_time = st.session_state.until_time_total
        
        new_chapter = st.number_input(
            "Until Chapter", min_value=0, value=current_chapter, 
            help="0 = all chapters", key="chapter_input"
        )
        
        c1, c2 = st.columns(2)
        with c1:
            new_minutes = st.number_input("Min", min_value=0, 
                value=current_time // 60, step=1, key="minutes_input")
        with c2:
            new_seconds = st.number_input("Sec", min_value=0, 
                value=current_time % 60, step=1, key="seconds_input")
        
        new_time_total = (new_minutes * 60) + new_seconds
        
        if new_chapter != current_chapter or new_time_total != current_time:
            st.warning("This will reset your chat!")
            if st.button("Apply Changes", type="primary", use_container_width=True):
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []
                st.session_state.until_chapter = new_chapter
                st.session_state.until_time_total = new_time_total
                st.rerun()

    # Chat header
    title = book.get('title', 'Untitled')
    author = clean_author_name(book.get('author', ''))
    
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)
    if author != "Unknown Author":
        st.caption(f"by {author}")
    
    st.divider()
    
    # Chat messages
    if st.session_state.messages:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-message">{message["content"]}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="assistant-message">{message["content"]}</div>', 
                           unsafe_allow_html=True)
                if "audio" in message:
                    for ref in message["audio"]:
                        if "url" in ref:
                            start_time = int(ref.get("start_time", 0))
                            st.audio(ref["url"], start_time=start_time)
                            st.caption(f"Chapter {ref.get('chapter_id')} at {start_time}s")
    else:
        st.markdown("""
            <div class="empty-chat">
                <p>Start asking questions about this book</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    # Chat input
    prompt = st.chat_input("Ask a question about this audiobook...")
    
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.rerun()

# Process chat response
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
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"Error: {response.status_code}"
                    })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": f"Connection failed: {e}"
                })
            
            st.rerun()

# ========================================================================
# PAGE: LIBRARY
# ========================================================================
elif page == "Library":
    st.session_state.current_page = "Library"
    
    # Page header
    render_page_header("Audiobook Library", show_refresh=True, refresh_key="refresh_library")
    
    books = fetch_books_from_api()
    
    if not books:
        st.info("No books found. Try refreshing or import a new book.")
    else:
        cols = st.columns(2, gap="large")
        for i, book in enumerate(books):
            with cols[i % 2]:
                book_title = book.get("title", "Untitled")
                clean_author = clean_author_name(book.get("author", ""))
                book_id = book.get("book_id", "")
                
                with st.container(border=True):
                    # Try to load image from GCS, fallback to placeholder
                    image_url = get_book_image_url(book_id)
                    
                    # Use HTML img tag with fallback via onerror
                    st.markdown(f"""
                        <div class="book-cover">
                            <img 
                                src="{image_url}" 
                                style="width: 100%; height: 100%; object-fit: cover; border-radius: 12px;"
                            />
                            <div class="book-cover-title" style="display: none; position: absolute; width: 100%; height: 100%; align-items: center; justify-content: center;">
                                {book_title}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Book info
                    st.markdown(f'<div class="book-title-text">{book_title}</div>', 
                               unsafe_allow_html=True)
                    st.markdown(f'<div class="book-author-text">{clean_author}</div>', 
                               unsafe_allow_html=True)
                    
                    # Chat button
                    if st.button("Start Chat", key=f"chat_{book['book_id']}", 
                                use_container_width=True, 
                                on_click=select_book_for_chat, args=(book,)):
                        st.rerun()

# ========================================================================
# PAGE: ADD NEW BOOK
# ========================================================================
elif page == "Add New Book":
    st.session_state.current_page = "Add New Book"
    
    render_page_header("Add New Book", subtitle="Upload an audiobook or import from Google Cloud Storage")
    
    tab_upload, tab_gcs = st.tabs(["Upload File", "Import from GCS"])
    
    with tab_upload:
        st.markdown("**Upload an audio file (MP3/WAV/M4A/OGG/FLAC) or a ZIP file containing audio chapters.**")
        
        with st.form("upload_form"):
            book_name = st.text_input("Book Name", placeholder="e.g., Harry Potter and the Sorcerer's Stone")
            author_name = st.text_input("Author Name (optional)", placeholder="e.g., J.K. Rowling")
            uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "m4a", "ogg", "flac", "zip"])
            submitted = st.form_submit_button("Upload Book", use_container_width=True)
            
            if submitted:
                if not book_name:
                    st.error("Please enter a book name.")
                elif not uploaded_file:
                    st.error("Please upload a file.")
                else:
                    processed_name = book_name.strip().lower().replace(" ", "_")
                    
                    with st.spinner("Uploading..."):
                        try:
                            files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                            data = {
                                "book_name": processed_name,
                                "author": author_name.strip() if author_name else ""
                            }
                            
                            response = requests.post(f"{API_URL}/upload-audio", files=files, data=data)
                            
                            if response.status_code == 200:
                                upload_result = response.json()
                                st.success(f"Successfully uploaded '{book_name}'!")
                                
                                st.info("Submitting processing job...")
                                
                                try:
                                    process_payload = {
                                        "folder_path": upload_result["folder_path"],
                                        "book_name": upload_result["book_name"],
                                        "author": author_name.strip() if author_name else "",
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
                                        
                                        if job_info.get('email_sent'):
                                            st.success(f"Confirmation email sent to {st.session_state.user_email}")
                                        else:
                                            email_error = job_info.get('email_error', 'Email service unavailable')
                                            st.warning(f"Could not send email notification: {email_error}")
                                        
                                        st.info("You'll receive an email when processing is complete!")
                                    else:
                                        st.error(f"Job submission failed: {process_response.status_code}")
                                        
                                except Exception as e:
                                    st.error(f"Processing connection error: {e}")
                            else:
                                st.error(f"Upload failed: {response.status_code}")
                        except Exception as e:
                            st.error(f"Connection error: {e}")
    
    with tab_gcs:
        st.markdown("**Import audiobook files already uploaded to Google Cloud Storage.**")
        st.info("Use this for large files that exceed the upload limit.")
        
        with st.form("gcs_import_form"):
            gcs_book_name = st.text_input("Book Name", 
                placeholder="e.g., Harry Potter and the Sorcerer's Stone", key="gcs_book")
            gcs_author_name = st.text_input("Author Name (optional)", 
                placeholder="e.g., J.K. Rowling", key="gcs_author")
            gcs_path = st.text_input("GCS Folder Path", 
                placeholder="gs://my-bucket/audiobooks/harry_potter/")
            
            if st.form_submit_button("Start Processing", use_container_width=True):
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
                                "author": gcs_author_name.strip() if gcs_author_name else "",
                                "target_tokens": 512,
                                "overlap_tokens": 50,
                                "model_size": "base",
                                "beam_size": 5,
                                "compute_type": "float32",
                                "user_email": st.session_state.user_email
                            }
                            
                            resp = requests.post(f"{API_URL}/process-audio", json=process_payload)
                            
                            if resp.status_code == 200:
                                job_info = resp.json()
                                st.success(f"Job Submitted! ID: {job_info.get('job_id')}")
                                st.info("Check 'My Activity' for progress.")
                            else:
                                st.error(f"Failed: {resp.status_code}")
                                
                        except Exception as e:
                            st.error(f"Connection error: {e}")

# ========================================================================
# PAGE: MY ACTIVITY
# ========================================================================
elif page == "My Activity":
    st.session_state.current_page = "My Activity"
    
    render_page_header("My Activity")
    
    tab1, tab2 = st.tabs(["Upload History", "Chat History"])
    
    with tab1:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.subheader("Your Upload Jobs")
        with col2:
            if st.button("Refresh", key="refresh_jobs", use_container_width=True):
                st.rerun()
        
        try:
            response = requests.get(f"{API_URL}/jobs/user/{st.session_state.user_email}")
            if response.status_code == 200:
                jobs = response.json()
                if not jobs:
                    st.info("No uploads yet. Go to 'Add New Book' to add an audiobook.")
                else:
                    for job in jobs:
                        with st.container(border=True):
                            c1, c2, c3 = st.columns([2, 1, 1])
                            
                            with c1:
                                st.markdown(f"**{job.get('book_name', 'Unknown')}**")
                                st.caption(f"ID: {job.get('job_id', 'N/A')}")
                            
                            with c2:
                                status = job.get('status', 'unknown')
                                if status == 'processing':
                                    st.warning("PROCESSING")
                                elif status == 'completed':
                                    st.success("COMPLETED")
                                elif status == 'failed':
                                    st.error("FAILED")
                                else:
                                    st.info(status.upper())
                            
                            with c3:
                                if status == 'processing':
                                    st.progress(job.get('progress', 0.0))
            else:
                st.warning("Unable to load history.")
        except:
            st.warning("Service temporarily unavailable")
    
    with tab2:
        col1, col2 = st.columns([5, 1])
        with col1:
            st.subheader("Your Chat History")
        with col2:
            if st.button("Refresh", key="refresh_chats", use_container_width=True):
                st.rerun()
        
        try:
            response = requests.get(f"{API_URL}/chat-history/{st.session_state.user_email}")
            if response.status_code == 200:
                history = response.json()
                if not history:
                    st.info("No chat history yet.")
                else:
                    for chat in history:
                        with st.container(border=True):
                            st.markdown(f"**{chat.get('book_title', 'Unknown')}**")
                            st.caption(chat.get('timestamp', ''))
                            st.write(f"Q: {chat.get('question', '')}")
                            with st.expander("View Answer"):
                                st.write(chat.get('answer', ''))
            else:
                st.info("Chat history coming soon!")
        except:
            st.info("Chat history coming soon!")

# ========================================================================
# PAGE: HEALTH CHECK
# ========================================================================
elif page == "Health Check":
    st.session_state.current_page = "Health Check"
    
    render_page_header("Service Health")
    
    if st.button("Check Health"):
        try:
            response = requests.get(f"{API_URL}/health")
            st.json(response.json())
            st.success("Backend is healthy!")
        except Exception as e:
            st.error(f"Failed to connect: {e}")

# ========================================================================
# PAGE: ADMIN DASHBOARD
# ========================================================================
elif page == "Admin Dashboard":
    st.session_state.current_page = "Admin Dashboard"
    
    if st.session_state.user_email not in ADMIN_EMAILS:
        st.error("Access Denied")
        st.stop()
    
    render_page_header("Admin Dashboard", show_refresh=True, refresh_key="refresh_admin")
    
    tab_stats, tab_mlflow = st.tabs(["System Metrics", "MLflow"])
    
    with tab_stats:
        with st.expander("Authorized Admins"):
            for email in ADMIN_EMAILS:
                st.code(email)
        
        try:
            with st.spinner("Loading..."):
                response = requests.get(f"{API_URL}/admin/stats")
                
            if response.status_code == 200:
                stats = response.json()
                db_stats = stats.get("database", {})
                job_stats = stats.get("jobs", {})
                books = stats.get("books", [])
                
                # Sync button
                if st.button("Sync from Cloud"):
                    with st.spinner("Syncing..."):
                        try:
                            sync_resp = requests.post(f"{API_URL}/admin/sync")
                            if sync_resp.status_code == 200:
                                st.success("Sync complete!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Sync failed")
                        except:
                            st.error("Connection failed")
                
                # Initialize view state if not present
                if "admin_view" not in st.session_state:
                    st.session_state.admin_view = "books"

                st.subheader("Overview")
                
                # Metrics as buttons for navigation
                m1, m2, m3, m4 = st.columns(4)
                
                with m1:
                    st.metric("Total Books", db_stats.get('total_books', 0))
                    if st.button("View Books", use_container_width=True):
                        st.session_state.admin_view = "books"
                        st.rerun()

                with m2:
                    st.metric("Active Jobs", job_stats.get('processing', 0))
                    if st.button("View Active", use_container_width=True):
                        st.session_state.admin_view = "processing"
                        st.rerun()

                with m3:
                    st.metric("Completed", job_stats.get('completed', 0))
                    if st.button("View Completed", use_container_width=True):
                        st.session_state.admin_view = "completed"
                        st.rerun()

                with m4:
                    st.metric("Failed", job_stats.get('failed', 0))
                    if st.button("View Failed", use_container_width=True):
                        st.session_state.admin_view = "failed"
                        st.rerun()
                
                st.divider()
                
                # Display content based on selection
                if st.session_state.admin_view == "books":
                    st.subheader("Books Library")
                    if books:
                        df = pd.DataFrame(books)
                        df['Status'] = df['chunk_count'].apply(
                            lambda x: '✅ Ready' if x > 0 else 'Processing')
                        
                        # Format dates
                        df = format_dataframe_dates(df)
                        
                        # Reorder columns for better view
                        display_cols = ['book_id', 'title', 'author', 'Status', 'chunk_count', 'created_at']
                        # Filter to only existing cols
                        display_cols = [c for c in display_cols if c in df.columns]
                        
                        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)
                    else:
                        st.info("No books found.")
                        
                elif st.session_state.admin_view == "processing":
                    st.subheader("Active Processing Jobs")
                    active_jobs = stats.get("active_jobs", [])
                    if active_jobs:
                        df = pd.DataFrame(active_jobs)
                        df = format_dataframe_dates(df)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No active jobs currently running.")
                        
                elif st.session_state.admin_view in ["completed", "failed"]:
                    status_title = st.session_state.admin_view.title()
                    st.subheader(f"{status_title} Jobs")
                    
                    with st.spinner(f"Fetching {st.session_state.admin_view} jobs..."):
                        try:
                            j_resp = requests.get(f"{API_URL}/admin/jobs?status={st.session_state.admin_view}")
                            if j_resp.status_code == 200:
                                jobs_data = j_resp.json().get("jobs", [])
                                if jobs_data:
                                    df = pd.DataFrame(jobs_data)
                                    df = format_dataframe_dates(df)
                                    st.dataframe(df, use_container_width=True, hide_index=True)
                                else:
                                    st.info(f"No {st.session_state.admin_view} jobs found.")
                            else:
                                st.error("Failed to fetch jobs")
                        except Exception as e:
                            st.error(f"Error fetching jobs: {e}")
                    
            else:
                st.error("Failed to fetch stats")
        except Exception as e:
            st.error(f"Error: {e}")
                    
    with tab_mlflow:
        st.subheader("MLflow Experiments")
        
        try:
            import mlflow
            
            mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5001")
            mlflow.set_tracking_uri(mlflow_uri)
            
            st.info(f"MLflow: {mlflow_uri}")
            st.link_button("Open MLflow", mlflow_uri)
            
        except Exception as e:
            st.error(f"MLflow error: {e}")