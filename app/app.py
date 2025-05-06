import streamlit as st
import sys
import os
import json
# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.chatbot import RealEstateChatbot

# Use Streamlit secrets for API key
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Set page config
st.set_page_config(
    page_title="AI Engineer Assignment - Gen AI + Agentic AI Challenge",
    page_icon="🏠",
    layout="centered"
)

# Initialize chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = RealEstateChatbot(api_key=openai_api_key)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "I'm real estate chatbot. How can I help you with buying or selling a property?"
    }]

# Custom CSS for better sidebar layout
st.markdown("""
<style>
    .sidebar .sidebar-content {
        padding-top: 1rem;
    }
    .sidebar .sidebar-content .block-container {
        padding-top: 0;
    }
    .sidebar h1 {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .sidebar h3 {
        font-size: 1.1rem;
        margin: 0.5rem 0;
    }
    .sidebar p {
        margin: 0.3rem 0;
    }
    .sidebar ul {
        margin: 0.3rem 0;
        padding-left: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar with assignment details
with st.sidebar:
    st.markdown(
        '''
        <style>
        .sidebar-section { margin-bottom: 0.4rem; }
        .sidebar-section-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 0.1rem; }
        .sidebar-section-sub { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.2rem; }
        .sidebar-section-list { font-size: 0.92rem; margin: 0.1rem 0 0.1rem 1rem; padding: 0; }
        .sidebar-section-list li { margin-bottom: 0.1rem; }
        .sidebar-section-hr { margin: 0.3rem 0; border: none; border-top: 1px solid #444; }
        .sidebar-thankyou {
            font-size: 1.5rem;
            font-weight: 900;
            color: #fff;
            text-align: center;
            margin-top: 1.2rem;
            margin-bottom: 0.2rem;
            letter-spacing: 0.5px;
        }
        </style>
        <div class="sidebar-section">
            <div class="sidebar-section-title">AI Engineer Assignment</div>
            <div class="sidebar-section-sub">Gen AI + Agentic AI Challenge</div>
        </div>
        <hr class="sidebar-section-hr">
        <div class="sidebar-section">
            <span class="sidebar-section-title">Company:</span> <span class="sidebar-section-sub">Commversion</span><br>
            <span class="sidebar-section-title">Candidate:</span>
            <ul class="sidebar-section-list">
                <li>Name: Harshal Honde</li>
                <li>Email: Harshalhonde50@gmail.com</li>
            </ul>
        </div>
        <hr class="sidebar-section-hr">
        <div class="sidebar-section">
            <div class="sidebar-section-title">Project Overview</div>
            <ul class="sidebar-section-list">
                <li>Intent understanding (buy/sell)</li>
                <li>New home and re-sale logic</li>
                <li>Input validation</li>
                <li>Postcode verification</li>
                <li>Context-aware responses</li>
            </ul>
        </div>
        <hr class="sidebar-section-hr">
        <div class="sidebar-section">
            <div class="sidebar-section-title">Technical Stack</div>
            <ul class="sidebar-section-list">
                <li>OpenRouter API (openai/gpt-4o)</li>
                <li>LangChain/Langgraph</li>
                <li>Streamlit</li>
                <li>Python</li>
                <li>Vector Store (Memory)</li>
            </ul>
        </div>
        <div class="sidebar-thankyou">Thank you for reviewing assignment</div>
        ''', unsafe_allow_html=True
    )
    if st.button("Reset Chat", help="Click to start a new conversation"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "I'm real estate chatbot. How can I help you with buying or selling a property?"
        }]
        st.rerun()

# Main content area
st.title("🏠 Real Estate Decision Assistant")

# Disclaimer expander
with st.expander("ℹ️ About This Demo"):
    st.caption(
        """This is a demo version of the real estate decision assistant. The system is designed to
        help users navigate through the home buying and selling process. Please note that this demo
        is limited to 10 interactions and may be unavailable if too many people use the service
        concurrently. Thank you for your understanding.
        """
    )

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("How can I help you with your real estate needs?")

if prompt:
    # Add user's message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get and display assistant response
    with st.chat_message("assistant"):
        try:
            response = st.session_state.chatbot.process_message(prompt)
            st.markdown(response["assistant_response"])
            st.session_state.messages.append(
                {"role": "assistant", "content": response["assistant_response"]}
            )
        except Exception as e:
            error_message = f"❌ Error: {str(e)}"
            st.markdown(error_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_message}
            ) 