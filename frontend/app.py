import streamlit as st
import requests
import os
import uuid

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8001")

st.set_page_config(page_title="AudioSeek", layout="wide")

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_book" not in st.session_state:
    st.session_state.selected_book = None

st.title("AudioSeek Interface")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Library", "Health Check"])

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

elif page == "Health Check":
    st.header("Service Health")
    if st.button("Check Health"):
        try:
            response = requests.get(f"{API_URL}/health")
            st.json(response.json())
        except Exception as e:
            st.error(f"Failed to connect to service: {e}")
