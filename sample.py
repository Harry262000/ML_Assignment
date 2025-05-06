"""
Real Estate Chatbot - Single File Implementation with LangChain
This file contains all the necessary components for the real estate chatbot:
- LangChain integration for better conversation management
- Vector store implementation (with in-memory fallback)
- Prompt templates and chains
- Chatbot core logic
- Streamlit UI
"""
import streamlit as st
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.memory import VectorStoreRetrieverMemory

# ============= Prompt Templates =============
INTENT_RECOGNITION_TEMPLATE = """
You are a real estate assistant. Analyze the following user message and determine their intent.
Possible intents are: BUY_HOME, SELL_HOME, GENERAL_QUERY, INVALID

User message: {message}

Intent:"""

RESPONSE_TEMPLATE = """
You are a real estate assistant. Follow this business logic for every conversation:

- If the user wants to BUY:
    - Ask for their name, phone, and email (one at a time, only if not already collected).
    - Ask for their budget.
        - If budget < 1 million: Politely say you only cater to properties above 1 million and provide the office number.
        - If budget >= 1 million: Ask for the postcode of interest.
            - If postcode is not covered: Politely say you don't cater to that postcode and provide the office number.
            - If postcode is covered: Assure a callback within 24 hours and ask if you can help with anything else.
                - If yes: Re-assist.
                - If no: Thank and close the chat.

- If the user wants to SELL:
    - Ask for their name, phone, email, and postcode (one at a time, only if not already collected).
        - If postcode is not covered: Politely say you don't cater to that postcode and provide the office number.
        - If postcode is covered: Assure a callback within 24 hours and ask if you can help with anything else.
            - If yes: Re-assist.
            - If no: Thank and close the chat.

- Never assist with renting.
- Always remember previously collected information and do not ask for it again.
- Use polite, professional language.
- The office number is 1300 111 222.

Assume the following postcodes are NOT covered: 0000, 9999, 1234. All others are covered.

Conversation so far:
{conversation_history}

User message: {message}

Assistant:"""

# ============= Vector Store Implementation =============
class InMemoryVectorStore:
    """Simple in-memory vector store implementation."""
    def __init__(self):
        self.documents = []
        self.metadatas = []
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        """Add documents to in-memory store."""
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
    
    def query(self, query_text: str, n_results: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Simple keyword-based search."""
        if not self.documents:
            return None
        return {
            "documents": self.documents[-n_results:],
            "metadatas": self.metadatas[-n_results:]
        }

class VectorStore:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        """Initialize the vector store."""
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions
            
            os.makedirs(persist_directory, exist_ok=True)
            
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            self.collection = self.client.get_or_create_collection(
                name="real_estate_chat",
                embedding_function=self.embedding_function
            )
            self.use_chroma = True
            
        except (ImportError, RuntimeError) as e:
            print(f"Warning: ChromaDB not available, using in-memory store: {e}")
            self.store = InMemoryVectorStore()
            self.use_chroma = False
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        """Add documents to the vector store."""
        if self.use_chroma:
            try:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=[str(i) for i in range(len(documents))]
                )
            except Exception as e:
                print(f"Warning: Failed to add documents to ChromaDB: {e}")
        else:
            self.store.add_documents(documents, metadatas)
    
    def query(self, query_text: str, n_results: int = 5) -> Optional[List[Dict[str, Any]]]:
        """Query the vector store."""
        if self.use_chroma:
            try:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results
                )
                return results
            except Exception as e:
                print(f"Warning: Failed to query ChromaDB: {e}")
                return None
        else:
            return self.store.query(query_text, n_results)

# ============= LangChain Integration =============
class LangChainRealEstateChatbot:
    def __init__(self, api_key: str, vector_store: Optional[VectorStore] = None):
        """Initialize the LangChain-powered chatbot."""
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model_name="x-ai/grok-3-mini-beta",
            temperature=0.7,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        # Initialize vector store
        self.vector_store = vector_store or VectorStore()
        
        # Initialize conversation memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize intent recognition chain
        self.intent_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["message"],
                template=INTENT_RECOGNITION_TEMPLATE
            )
        )
        
        # Initialize main conversation chain
        self.conversation_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
        
        # Initialize response generation chain
        self.response_chain = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate(
                input_variables=["intent", "message", "conversation_history"],
                template=RESPONSE_TEMPLATE
            )
        )
        
        # Initialize text splitter for document processing
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def detect_intent(self, message: str) -> str:
        """Detect the user's intent using LangChain."""
        try:
            intent = self.intent_chain.predict(message=message).strip().upper()
            valid_intents = ["BUY_HOME", "SELL_HOME", "GENERAL_QUERY", "INVALID"]
            
            if intent not in valid_intents:
                return "GENERAL_QUERY"
            
            return intent
            
        except Exception as e:
            print(f"Error detecting intent: {e}")
            return "GENERAL_QUERY"
    
    def generate_response(
        self,
        message: str,
        intent: str,
        conversation_history: List[Dict[str, Any]]
    ) -> str:
        """Generate a response using LangChain."""
        try:
            # Format conversation history
            history_context = "\n".join([
                f"User: {entry['user_message']}\nAssistant: {entry['assistant_response']}"
                for entry in conversation_history[-5:]
            ])
            
            # Generate response using LangChain
            response = self.response_chain.predict(
                intent=intent,
                message=message,
                conversation_history=history_context
            )
            
            # Update conversation memory
            self.memory.save_context(
                {"input": message},
                {"output": response}
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try again."
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message using LangChain."""
        # Detect intent
        intent = self.detect_intent(message)
        
        # Generate response
        response = self.generate_response(message, intent, self.memory.chat_memory.messages)
        
        # Store in vector store
        if self.vector_store:
            self.vector_store.add_documents(
                documents=[message, response],
                metadatas=[{"type": "user", "intent": intent}, {"type": "assistant", "intent": intent}]
            )
        
        return {
            "timestamp": datetime.now(),
            "user_message": message,
            "assistant_response": response,
            "intent": intent
        }

# ============= Streamlit UI =============
def main():
    # Set page config
    st.set_page_config(
        page_title="Real Estate Assistant",
        page_icon="🏠",
        layout="centered"
    )
    
    # Initialize chatbot
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = LangChainRealEstateChatbot(api_key=st.secrets["OPENAI_API_KEY"])
    
    # Initialize session state for messages
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": "I'm real estate chatbot. How can I help you with buying or selling a property?"
        }]
    
    # Main content area
    st.title("🏠 Real Estate Decision Assistant")
    
    # Sidebar with project information
    with st.sidebar:
        st.header("About This Project")
        st.markdown("""
        ### Real Estate Decision Assistant
        This chatbot helps users make informed decisions about buying or selling property.
        
        #### Technical Stack:
        - LangChain for conversation management
        - OpenRouter API (grok-3-mini-beta)
        - Vector Store (Memory)
        - Streamlit UI
        
        #### Features:
        - Intent recognition
        - Contextual responses
        - Conversation memory
        - Vector-based search
        """)
    
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

if __name__ == "__main__":
    main()