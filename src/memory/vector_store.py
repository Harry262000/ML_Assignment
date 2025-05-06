"""
Vector store implementation using Chroma.
"""
from typing import List, Dict, Any, Optional
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

class VectorStore:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector store
        """
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client with default embedding function
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Use default embedding function
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="real_estate_chat",
                embedding_function=self.embedding_function
            )
            
        except Exception as e:
            print(f"Warning: Failed to initialize vector store: {e}")
            self.client = None
            self.collection = None
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
        """
        if not self.collection:
            return
            
        try:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=[str(i) for i in range(len(documents))]
            )
        except Exception as e:
            print(f"Warning: Failed to add documents to vector store: {e}")
    
    def query(self, query_text: str, n_results: int = 5) -> Optional[List[Dict[str, Any]]]:
        """
        Query the vector store.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            List of results with documents and metadata, or None if query fails
        """
        if not self.collection:
            return None
            
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Warning: Failed to query vector store: {e}")
            return None 