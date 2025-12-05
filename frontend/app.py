import streamlit as st
import requests
import os
import uuid
from dotenv import load_dotenv
import streamlit_oauth as oauth
import time

# Load env vars
load_dotenv(".env.local")

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8001")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

st.set_page_config(page_title="AudioSeek", layout="wide")

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_book" not in st.session_state:
    st.session_state.selected_book = None

if "user_email" not in st.session_state:
    st.session_state.user_email = None

st.title("AudioSeek Interface")

# ----------------------------------------------------------------
# AUTHENTICATION
# ----------------------------------------------------------------
if not st.session_state.user_email:
    st.info("Please sign in to continue.")
    
    if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
        try:
            result = oauth.OAuth2Component(
                client_id=GOOGLE_CLIENT_ID,
                client_secret=GOOGLE_CLIENT_SECRET,
                auth_uri="https://accounts.google.com/o/oauth2/auth",
                token_uri="https://oauth2.googleapis.com/token",
                auth_provider_x509_cert_url="https://www.googleapis.com/oauth2/v1/certs",
                redirect_uri=GOOGLE_REDIRECT_URI,
                scope="openid email profile",
            ).authorize_button(
                name="Sign in with Google",
                icon="https://www.google.com/favicon.ico",
                redirect_uri=GOOGLE_REDIRECT_URI,
                key="google_auth"
            )
            
            if result and "token" in result:
                # Decode ID token to get email (simplified, ideally verify signature)
                # For now we trust the token returned by the direct OAuth flow
                # result["id_token"] contains the JWT
                import base64
                import json
                
                # Simple JWT decode without verification (verification should happen on backend)
                id_token = result.get("id_token")
                if id_token:
                    # JWT is header.payload.signature
                    parts = id_token.split(".")
                    if len(parts) > 1:
                        payload = parts[1]
                        # Pad base64
                        payload += "=" * ((4 - len(payload) % 4) % 4)
                        decoded = base64.urlsafe_b64decode(payload)
                        user_info = json.loads(decoded)
                        st.session_state.user_email = user_info.get("email")
                        st.session_state.user_name = user_info.get("name")
                        st.success(f"Welcome, {st.session_state.user_name}!")
                        st.rerun()
        except Exception as e:
            st.error(f"Authentication error: {e}")
    else:
        st.warning("Google OAuth credentials not configured.")
        
    st.stop() # Stop execution until logged in

# ----------------------------------------------------------------
# MAIN APP (Only reachable if logged in)
# ----------------------------------------------------------------

# Sidebar
st.sidebar.header(f"User: {st.session_state.user_email}")
if st.sidebar.button("Logout"):
    st.session_state.user_email = None
    st.rerun()

st.sidebar.divider()
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Library", "My Uploads", "Add New Book", "Health Check"])

if page == "Library":
    st.header("Audiobook Library")
    
    # Fetch books
    try:
        response = requests.get(f"{API_URL}/books")
        if response.status_code == 200:
            books = response.json()
        else:
            st.error(f"Failed to fetch books: {response.status_code}")
            books = []
    except Exception as e:
        st.error(f"Connection error: {e}")
        books = []

    if not books:
        st.info("No books found in the library.")
    else:
        # Display books in a grid or list
        cols = st.columns(3)
        for i, book in enumerate(books):
            with cols[i % 3]:
                with st.container(border=True):
                    st.subheader(book.get('title', 'Unknown Title'))
                    st.write(f"**Author:** {book.get('author', 'Unknown')}")
                    if st.button("Chat with this book", key=book['book_id']):
                        st.session_state.selected_book = book
                        st.session_state.messages = [] # Reset chat on book switch
                        st.rerun()

    # Chat Interface
    if st.session_state.selected_book:
        st.divider()
        book = st.session_state.selected_book
        st.subheader(f"Chatting with: {book.get('title')}")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about this book..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        payload = {
                            "query": prompt,
                            "book_id": book['book_id'],
                            "session_id": st.session_state.session_id
                        }
                        response = requests.post(f"{API_URL}/qa/ask", json=payload)
                        
                        if response.status_code == 200:
                            result = response.json()
                            answer = result.get("answer", "No answer provided.")
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            error_msg = f"Error: {response.status_code} - {response.text}"
                            st.error(error_msg)
                            st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    except Exception as e:
                        error_msg = f"Connection failed: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

elif page == "My Uploads":
    st.header("My Upload Jobs")
    if st.button("Refresh Status"):
        st.rerun()
        
    try:
        response = requests.get(f"{API_URL}/jobs/user/{st.session_state.user_email}")
        if response.status_code == 200:
            jobs = response.json()
            if not jobs:
                st.info("No uploads found.")
            else:
                for job in jobs:
                    with st.container(border=True):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.subheader(job.get('book_name'))
                            st.write(f"Job ID: `{job.get('job_id')}`")
                            st.write(f"Status: **{job.get('status')}**")
                            if job.get('message'):
                                st.info(job.get('message'))
                        with col2:
                            if job.get('status') == 'processing':
                                st.progress(job.get('progress', 0.0))
                            elif job.get('status') == 'completed':
                                st.success("Done")
                            elif job.get('status') == 'failed':
                                st.error("Failed")
        else:
            st.error(f"Failed to fetch jobs: {response.status_code}")
    except Exception as e:
        st.error(f"Connection error: {e}")

elif page == "Add New Book":
    st.header("Request New Book")
    st.write("Upload an audio file (MP3/WAV) or a ZIP file containing audio chapters.")

    with st.form("upload_form"):
        book_name = st.text_input("Book Name (e.g., 'pride_and_prejudice')")
        uploaded_file = st.file_uploader("Choose a file", type=["mp3", "wav", "zip"])
        submitted = st.form_submit_button("Upload Book")

        if submitted:
            if not book_name:
                st.error("Please enter a book name.")
            elif not uploaded_file:
                st.error("Please upload a file.")
            else:
                with st.spinner("Uploading..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        data = {"book_name": book_name}
                        
                        response = requests.post(f"{API_URL}/upload-audio", files=files, data=data)
                        
                        if response.status_code == 200:
                            upload_result = response.json()
                            st.success(f"Successfully uploaded '{book_name}'!")
                            
                            # Trigger Async Processing
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
                                    st.info("You can track the progress in the 'My Uploads' tab.")
                                else:
                                    st.error(f"Submission failed: {process_response.status_code} - {process_response.text}")
                                    
                            except Exception as e:
                                st.error(f"Processing connection error: {e}")

                        else:
                            st.error(f"Upload failed: {response.status_code} - {response.text}")
                    except Exception as e:
                        st.error(f"Connection error: {e}")

elif page == "Health Check":
    st.header("Service Health")
    if st.button("Check Health"):
        try:
            response = requests.get(f"{API_URL}/health")
            st.json(response.json())
        except Exception as e:
            st.error(f"Failed to connect to service: {e}")
