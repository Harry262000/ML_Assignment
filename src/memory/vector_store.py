from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persist_directory: str = ":memory:"):
        """
        Initialize the vector store. Default to in-memory.
        
        Args:
            persist_directory: Directory to persist the vector store (defaults to in-memory)
        """
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection("real_estate_chat")
    
    def add_documents(self, documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            metadatas: List of metadata dictionaries
        """
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=[str(i) for i in range(len(documents))]
        )
    
    def query(self, query_text: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector store.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            
        Returns:
            List of results with documents and metadata
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results
        )
        return results
