"""
Vector store implementation using Chroma or in-memory fallback.
"""
from typing import List, Dict, Any, Optional
import os

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
        # Return most recent documents as fallback
        return {
            "documents": self.documents[-n_results:],
            "metadatas": self.metadatas[-n_results:]
        }

class VectorStore:
    def __init__(self, persist_directory: str = "data/chroma_db"):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory to persist the vector store
        """
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions
            
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
            self.use_chroma = True
            
        except (ImportError, RuntimeError) as e:
            print(f"Warning: ChromaDB not available, using in-memory store: {e}")
            self.store = InMemoryVectorStore()
            self.use_chroma = False
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
        """
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
        """
        Query the vector store.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            List of results with documents and metadata, or None if query fails
        """
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
