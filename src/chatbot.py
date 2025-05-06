from typing import Dict, List, Any, Optional
from datetime import datetime
from openai import OpenAI
from src.memory.vector_store import VectorStore
from .prompts.templates import (
    INTENT_RECOGNITION_TEMPLATE,
    RESPONSE_TEMPLATE
)

class RealEstateChatbot:
    def __init__(self, api_key: str, vector_store: Optional[VectorStore] = None):
        """
        Initialize the chatbot.
        
        Args:
            api_key: OpenAI API key
            vector_store: Optional vector store for conversation memory (defaults to in-memory storage)
        """
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        # If no vector store is provided, use the in-memory vector store
        self.vector_store = vector_store or VectorStore()  # This will use in-memory by default
        self.conversation_history = []
        self.model = "x-ai/grok-3-mini-beta"
    
    def detect_intent(self, message: str) -> str:
        """
        Detect the user's intent from their message.
        
        Args:
            message: User's message
            
        Returns:
            Detected intent (BUY_HOME, SELL_HOME, GENERAL_QUERY, or INVALID)
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": INTENT_RECOGNITION_TEMPLATE},
                          {"role": "user", "content": message}],
                temperature=0.3,
                max_tokens=50,
                top_p=1,
            )
            
            intent = completion.choices[0].message.content.strip().upper()
            valid_intents = ["BUY_HOME", "SELL_HOME", "GENERAL_QUERY", "INVALID"]
            
            # Ensure the intent is valid
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
        """
        Generate a response based on the user's message and intent.
        
        Args:
            message: User's message
            intent: Detected intent
            conversation_history: List of previous messages
            
        Returns:
            Generated response
        """
        try:
            # Format conversation history for context (only last 5 messages)
            history_context = "\n".join([f"User: {entry['user_message']}\nAssistant: {entry['assistant_response']}"
                                        for entry in conversation_history[-5:]])  # Last 5 messages for context
            
            # Create the prompt with context
            prompt = RESPONSE_TEMPLATE.format(
                intent=intent,
                message=message,
                conversation_history=history_context
            )
            
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": prompt},
                          {"role": "user", "content": message}],
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
            )
            
            return completion.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble processing your request. Please try again."
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a user message and generate a response.
        
        Args:
            message: User's message
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Detect intent
        intent = self.detect_intent(message)
        
        # Generate response
        response = self.generate_response(message, intent, self.conversation_history)
        
        # Store the conversation in memory
        conversation_entry = {
            "timestamp": datetime.now(),
            "user_message": message,
            "assistant_response": response,
            "intent": intent
        }
        self.conversation_history.append(conversation_entry)
        
        # Store the conversation in the vector store if available
        if self.vector_store:
            self.vector_store.add_documents(
                documents=[message, response],
                metadatas=[{"type": "user", "intent": intent}, {"type": "assistant", "intent": intent}]
            )
        
        return conversation_entry 