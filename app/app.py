import streamlit as st
import sys
import os
from datetime import datetime

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.chatbot import RealEstateChatbot
from src.memory.vector_store import VectorStore

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

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message when first loading
    st.session_state.messages.append({
        "role": "assistant",
        "content": "I'm real estate chatbot (testing) ? How can I help you?"
    })

if "max_messages" not in st.session_state:
    # Counting both user and assistant messages, so 10 rounds of conversation
    st.session_state.max_messages = 20

# Initialize show_suggestions in session state
if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True

# Add to session state: track which field to ask next and store user info
if "user_info" not in st.session_state:
    st.session_state.user_info = {
        "name": None,
        "email": None,
        "phone": None,
        "budget": None,
        "postcode": None,
        "home_type": None,
        "intent": None,
        "step": None  # Tracks which field to ask next
    }

def get_next_field(intent, user_info):
    # Define the order of fields for each intent
    if intent == "buy":
        fields = ["name", "email", "phone", "budget", "postcode", "home_type"]
    elif intent == "sell":
        fields = ["name", "email", "phone", "postcode"]
    else:
        return None
    for field in fields:
        if not user_info.get(field):
            return field
    return None

def get_field_prompt(field):
    prompts = {
        "name": "What is your name?",
        "email": "May I have your email address, please?",
        "phone": "May I have your phone number, please?",
        "budget": "What is your budget?",
        "postcode": "What is your postcode?",
        "home_type": "Are you interested in a new home or a resale home?"
    }
    return prompts.get(field, "")

# Add a function to reset the chat
def reset_chat():
    st.session_state.messages = [{
        "role": "assistant",
        "content": "I'm real estate chatbot (testing) ? How can I help you?"
    }]
    st.session_state.show_suggestions = True

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
                <li>LangChain/Langgraph</li>
                <li>Groq (Mixtral-8x7b)</li>
                <li>Streamlit</li>
                <li>Python</li>
            </ul>
        </div>
        <div class="sidebar-thankyou">Thank you for reviewing assignment</div>
        ''', unsafe_allow_html=True
    )
    if st.button("Reset Chat", help="Click to start a new conversation"):
        reset_chat()
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

# Main chat logic
if len(st.session_state.messages) >= st.session_state.max_messages:
    st.info(
        """Notice: The maximum message limit for this demo version has been reached. We value your interest!
        We encourage you to experience further interactions by building your own application with instructions
        from Streamlit's [Build a basic LLM chat app](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)
        tutorial. Thank you for your understanding."""
    )
else:
    # Always show chat input
    prompt = st.chat_input("How can I help you with your real estate needs?")
    
    if prompt:
        # Always add user's message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Prepare conversation history for the LLM
        conversation_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ]

        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                # Pass the conversation history to the chatbot
                response = st.session_state.chatbot.process_message(prompt)
                st.markdown(response["assistant_response"])
                st.session_state.messages.append(
                    {"role": "assistant", "content": response["assistant_response"]}
                )
                if "conversation_history" not in st.session_state:
                    st.session_state.conversation_history = []
                st.session_state.conversation_history.append(response)
            except Exception as e:
                st.session_state.max_messages = len(st.session_state.messages)
                error_message = f"❌ Error: {str(e)}"
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                ) 