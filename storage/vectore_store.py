"""
Vector Store for Semantic Search

This module provides LangChain vector store functionality for semantic
search over domain knowledge, edge cases, and Z3 theories.

Key improvements:
- Semantic search instead of keyword matching
- Automatic embedding generation
- Persistent storage
- Easy to query and update

New feature: Adds vector search capability to the system
"""

from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import settings
from storage.database import DatabaseManager


class KnowledgeBase:
    """
    Vector store for semantic search over knowledge base.
    
    Provides semantic search over:
    - Domain knowledge
    - Edge case patterns
    - Z3 theory examples
    
    Example:
        >>> kb = KnowledgeBase()
        >>> kb.initialize()
        >>> results = kb.search("array sorting edge cases", k=5)
        >>> for doc in results:
        ...     print(doc.page_content)
    """
    
    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embedding_model: Optional[str] = None
    ):
        """
        Initialize the knowledge base.
        
        Args:
            persist_directory: Where to store the vector database
            embedding_model: OpenAI embedding model to use
        """
        self.persist_directory = persist_directory or settings.vector_store_dir
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.embedding_model = embedding_model or settings.embedding_model
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=settings.openai_api_key
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len
        )
        
        self.vector_store: Optional[Chroma] = None
        self.db_manager = DatabaseManager()
    
    def initialize(self, force_rebuild: bool = False) -> None:
        """
        Initialize the vector store, loading from existing or building new.
        
        Args:
            force_rebuild: If True, rebuild from scratch even if exists
            
        Example:
            >>> kb = KnowledgeBase()
            >>> kb.initialize()
        """
        # Check if vector store already exists
        if not force_rebuild and self._vector_store_exists():
            print("Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings
            )
            print(f"✅ Loaded vector store with {self.vector_store._collection.count()} documents")
        else:
            print("Building new vector store from database...")
            self._build_vector_store()
            print(f"✅ Built vector store with {self.vector_store._collection.count()} documents")
    
    def _vector_store_exists(self) -> bool:
        """Check if a persisted vector store exists."""
        chroma_db_path = self.persist_directory / "chroma.sqlite3"
        return chroma_db_path.exists()
    
    def _build_vector_store(self) -> None:
        """Build vector store from database content."""
        documents = []
        
        # Load domain knowledge
        try:
            domain_knowledge = self.db_manager.query_domain_knowledge()
            for entry in domain_knowledge:
                doc_text = f"""
Domain: {entry.get('domain', 'Unknown')}
Patterns: {entry.get('patterns', '')}
Examples: {entry.get('examples', '')}
Common Ambiguities: {entry.get('common_ambiguities', '')}
Edge Cases: {entry.get('edge_cases', '')}
"""
                documents.append(Document(
                    page_content=doc_text.strip(),
                    metadata={
                        'source': 'domain_knowledge',
                        'domain': entry.get('domain', 'Unknown')
                    }
                ))
        except Exception as e:
            print(f"⚠️  Could not load domain knowledge: {e}")
        
        # Load Z3 theories
        try:
            z3_theories = self.db_manager.query_z3_theories()
            for entry in z3_theories:
                doc_text = f"""
Theory Type: {entry.get('theory_type', 'Unknown')}
Description: {entry.get('description', '')}
Example Code: {entry.get('example_code', '')}
Use Cases: {entry.get('use_cases', '')}
"""
                documents.append(Document(
                    page_content=doc_text.strip(),
                    metadata={
                        'source': 'z3_theory',
                        'theory_type': entry.get('theory_type', 'Unknown')
                    }
                ))
        except Exception as e:
            print(f"⚠️  Could not load Z3 theories: {e}")
        
        if not documents:
            print("⚠️  No documents found in database. Creating empty vector store.")
            # Create empty vector store
            self.vector_store = Chroma(
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
        else:
            # Split documents into chunks
            split_docs = self.text_splitter.split_documents(documents)
            print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
            
            # Create vector store
            self.vector_store = Chroma.from_documents(
                documents=split_docs,
                embedding=self.embeddings,
                persist_directory=str(self.persist_directory)
            )
            self.vector_store.persist()
    
    def search(
        self,
        query: str,
        k: int = 5,
        filter_by: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Semantic search over knowledge base.
        
        Args:
            query: Search query
            k: Number of results to return
            filter_by: Optional metadata filters
            
        Returns:
            List of relevant documents
            
        Example:
            >>> kb = KnowledgeBase()
            >>> kb.initialize()
            >>> results = kb.search("sorting algorithms edge cases", k=3)
            >>> for doc in results:
            ...     print(doc.metadata['domain'])
        """
        if self.vector_store is None:
            self.initialize()
        
        if filter_by:
            return self.vector_store.similarity_search(
                query,
                k=k,
                filter=filter_by
            )
        else:
            return self.vector_store.similarity_search(query, k=k)
    
    def search_with_scores(
        self,
        query: str,
        k: int = 5
    ) -> List[tuple[Document, float]]:
        """
        Semantic search with similarity scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
            
        Example:
            >>> results = kb.search_with_scores("array bounds checking", k=3)
            >>> for doc, score in results:
            ...     print(f"Score: {score:.3f} - {doc.page_content[:50]}")
        """
        if self.vector_store is None:
            self.initialize()
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def search_domain_knowledge(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        Search only domain knowledge.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of domain knowledge documents
        """
        return self.search(
            query=query,
            k=k,
            filter_by={'source': 'domain_knowledge'}
        )
    
    def search_z3_theories(
        self,
        query: str,
        k: int = 5
    ) -> List[Document]:
        """
        Search only Z3 theories.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of Z3 theory documents
        """
        return self.search(
            query=query,
            k=k,
            filter_by={'source': 'z3_theory'}
        )
    
    def add_documents(
        self,
        documents: List[Document]
    ) -> None:
        """
        Add new documents to the vector store.
        
        Args:
            documents: Documents to add
            
        Example:
            >>> kb = KnowledgeBase()
            >>> kb.initialize()
            >>> new_doc = Document(
            ...     page_content="New edge case information",
            ...     metadata={'source': 'user_input', 'domain': 'custom'}
            ... )
            >>> kb.add_documents([new_doc])
        """
        if self.vector_store is None:
            self.initialize()
        
        # Split documents
        split_docs = self.text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vector_store.add_documents(split_docs)
        self.vector_store.persist()
        
        print(f"✅ Added {len(split_docs)} document chunks")
    
    def rebuild(self) -> None:
        """
        Rebuild the vector store from scratch.
        
        Example:
            >>> kb = KnowledgeBase()
            >>> kb.rebuild()
        """
        self.initialize(force_rebuild=True)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def search_knowledge(query: str, k: int = 5) -> List[str]:
    """
    Convenience function for quick semantic search.
    
    Args:
        query: What to search for
        k: Number of results
        
    Returns:
        List of relevant text snippets
        
    Example:
        >>> results = search_knowledge("array sorting edge cases", k=3)
        >>> for result in results:
        ...     print(result)
    """
    kb = KnowledgeBase()
    kb.initialize()
    docs = kb.search(query, k=k)
    return [doc.page_content for doc in docs]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VECTOR STORE - EXAMPLE USAGE")
    print("=" * 70)
    
    # Example 1: Initialize and search
    print("\n🔍 Example 1: Initialize and search")
    print("-" * 70)
    
    kb = KnowledgeBase()
    kb.initialize()
    
    query = "sorting algorithms"
    results = kb.search(query, k=3)
    
    print(f"Query: '{query}'")
    print(f"Found {len(results)} results:")
    for i, doc in enumerate(results):
        print(f"\n{i+1}. Source: {doc.metadata.get('source', 'Unknown')}")
        print(f"   Preview: {doc.page_content[:100]}...")
    
    # Example 2: Search with scores
    print("\n📊 Example 2: Search with similarity scores")
    print("-" * 70)
    
    query = "edge cases"
    results_with_scores = kb.search_with_scores(query, k=3)
    
    print(f"Query: '{query}'")
    for i, (doc, score) in enumerate(results_with_scores):
        print(f"\n{i+1}. Similarity: {score:.3f}")
        print(f"   {doc.page_content[:80]}...")
    
    # Example 3: Filter by source
    print("\n🎯 Example 3: Search only domain knowledge")
    print("-" * 70)
    
    domain_results = kb.search_domain_knowledge("array operations", k=2)
    print(f"Found {len(domain_results)} domain knowledge results")
    
    # Example 4: Add custom document
    print("\n➕ Example 4: Add custom document")
    print("-" * 70)
    
    custom_doc = Document(
        page_content="Custom edge case: Always check for null pointers before dereferencing",
        metadata={'source': 'user_input', 'domain': 'memory_safety'}
    )
    
    kb.add_documents([custom_doc])
    print("✅ Added custom document")
    
    print("\n" + "=" * 70)
    print("✅ EXAMPLES COMPLETED")
    print("=" * 70)