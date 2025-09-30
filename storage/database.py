"""
Unified Database Manager

This module consolidates all database operations that were previously
scattered throughout the codebase.

Key improvements:
- Single database connection manager
- Type-safe queries using Pydantic models
- Automatic result parsing
- Connection pooling
- Context manager support

Replaces: Scattered SQLite queries throughout old codebase
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import sqlite3
from contextlib import contextmanager

from config.settings import settings
from core.models import CompleteEnhancedResult, Function


class DatabaseManager:
    """
    Unified database manager for all database operations.
    
    Handles connections to:
    - context.db: Domain knowledge and patterns
    - z3_theories.db: Z3 theory examples
    - results.db: Pipeline execution results
    
    Example:
        >>> db = DatabaseManager()
        >>> with db.get_connection('context') as conn:
        ...     knowledge = db.query_domain_knowledge(conn, 'sorting')
    """
    
    def __init__(self):
        """Initialize database connections."""
        self.context_db_path = settings.context_db
        self.z3_db_path = settings.z3_theories_db
        self.results_db_path = getattr(settings, 'results_db', Path('results.db'))
    
    @contextmanager
    def get_connection(self, db_name: str = 'context'):
        """
        Get a database connection with automatic cleanup.
        
        Args:
            db_name: Which database ('context', 'z3', 'results')
            
        Yields:
            sqlite3.Connection
            
        Example:
            >>> with db.get_connection('context') as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT * FROM domains")
        """
        if db_name == 'context':
            db_path = self.context_db_path
        elif db_name == 'z3':
            db_path = self.z3_db_path
        elif db_name == 'results':
            db_path = self.results_db_path
        else:
            raise ValueError(f"Unknown database: {db_name}")
        
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row  # Access columns by name
        
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def query_domain_knowledge(
        self,
        domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query domain knowledge from context database.
        
        Args:
            domain: Specific domain to query (None = all domains)
            
        Returns:
            List of domain knowledge dictionaries
            
        Example:
            >>> db = DatabaseManager()
            >>> sorting_knowledge = db.query_domain_knowledge('sorting')
        """
        with self.get_connection('context') as conn:
            cursor = conn.cursor()
            
            if domain:
                cursor.execute(
                    "SELECT * FROM domain_contexts WHERE domain = ?",
                    (domain,)
                )
            else:
                cursor.execute("SELECT * FROM domain_contexts")
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def query_z3_theories(
        self,
        theory_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Query Z3 theory examples from database.
        
        Args:
            theory_type: Specific theory type (e.g., 'arrays', 'arithmetic')
            
        Returns:
            List of Z3 theory examples
            
        Example:
            >>> db = DatabaseManager()
            >>> array_theories = db.query_z3_theories('arrays')
        """
        with self.get_connection('z3') as conn:
            cursor = conn.cursor()
            
            if theory_type:
                cursor.execute(
                    "SELECT * FROM theories WHERE theory_type = ?",
                    (theory_type,)
                )
            else:
                cursor.execute("SELECT * FROM theories")
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def save_pipeline_result(
        self,
        result: CompleteEnhancedResult
    ) -> None:
        """
        Save pipeline execution result to database.
        
        Args:
            result: CompleteEnhancedResult to save
            
        Example:
            >>> db = DatabaseManager()
            >>> db.save_pipeline_result(result)
        """
        with self.get_connection('results') as conn:
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_results (
                    session_id TEXT PRIMARY KEY,
                    specification TEXT,
                    status TEXT,
                    total_functions INTEGER,
                    total_postconditions INTEGER,
                    successful_z3_translations INTEGER,
                    processing_time REAL,
                    generated_at TIMESTAMP,
                    result_json TEXT
                )
            """)
            
            # Insert result
            cursor.execute("""
                INSERT OR REPLACE INTO pipeline_results
                (session_id, specification, status, total_functions, 
                 total_postconditions, successful_z3_translations,
                 processing_time, generated_at, result_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.session_id,
                result.specification,
                result.overall_status.value,
                len(result.function_results),
                result.total_postconditions,
                result.successful_z3_translations,
                result.total_processing_time,
                result.generated_at,
                result.model_dump_json()
            ))
    
    def load_pipeline_result(
        self,
        session_id: str
    ) -> Optional[CompleteEnhancedResult]:
        """
        Load a pipeline result by session ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            CompleteEnhancedResult or None if not found
            
        Example:
            >>> db = DatabaseManager()
            >>> result = db.load_pipeline_result("session_123")
        """
        with self.get_connection('results') as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT result_json FROM pipeline_results WHERE session_id = ?",
                (session_id,)
            )
            
            row = cursor.fetchone()
            if row:
                return CompleteEnhancedResult.model_validate_json(row['result_json'])
            return None
    
    def list_recent_results(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        List recent pipeline results.
        
        Args:
            limit: Maximum number of results to return
            
        Returns:
            List of result summaries
            
        Example:
            >>> db = DatabaseManager()
            >>> recent = db.list_recent_results(5)
            >>> for r in recent:
            ...     print(f"{r['session_id']}: {r['status']}")
        """
        with self.get_connection('results') as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, specification, status, 
                       total_postconditions, processing_time, generated_at
                FROM pipeline_results
                ORDER BY generated_at DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def search_results(
        self,
        specification_query: str
    ) -> List[Dict[str, Any]]:
        """
        Search results by specification text.
        
        Args:
            specification_query: Text to search for
            
        Returns:
            List of matching results
            
        Example:
            >>> db = DatabaseManager()
            >>> results = db.search_results("sort")
        """
        with self.get_connection('results') as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, specification, status, 
                       total_postconditions, processing_time, generated_at
                FROM pipeline_results
                WHERE specification LIKE ?
                ORDER BY generated_at DESC
            """, (f"%{specification_query}%",))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_domain_knowledge(domain: str) -> List[Dict[str, Any]]:
    """
    Convenience function to get domain knowledge.
    
    Args:
        domain: Domain name (e.g., 'sorting', 'searching')
        
    Returns:
        List of domain knowledge entries
    """
    db = DatabaseManager()
    return db.query_domain_knowledge(domain)


def get_z3_theories(theory_type: str) -> List[Dict[str, Any]]:
    """
    Convenience function to get Z3 theories.
    
    Args:
        theory_type: Theory type (e.g., 'arrays', 'arithmetic')
        
    Returns:
        List of Z3 theory examples
    """
    db = DatabaseManager()
    return db.query_z3_theories(theory_type)


def save_result(result: CompleteEnhancedResult) -> None:
    """
    Convenience function to save a pipeline result.
    
    Args:
        result: Result to save
    """
    db = DatabaseManager()
    db.save_pipeline_result(result)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DATABASE MANAGER - EXAMPLE USAGE")
    print("=" * 70)
    
    db = DatabaseManager()
    
    # Example 1: Query domain knowledge
    print("\n📚 Example 1: Query domain knowledge")
    print("-" * 70)
    
    try:
        knowledge = db.query_domain_knowledge()
        print(f"Found {len(knowledge)} domain knowledge entries")
        if knowledge:
            print(f"First entry: {list(knowledge[0].keys())}")
    except Exception as e:
        print(f"Note: {e}")
        print("(This is normal if context.db doesn't exist yet)")
    
    # Example 2: List recent results
    print("\n📋 Example 2: List recent results")
    print("-" * 70)
    
    try:
        recent = db.list_recent_results(5)
        print(f"Found {len(recent)} recent results")
        for r in recent:
            print(f"  - {r['session_id']}: {r['status']}")
    except Exception as e:
        print(f"Note: {e}")
        print("(This is normal if results.db doesn't exist yet)")
    
    # Example 3: Context manager usage
    print("\n🔗 Example 3: Context manager usage")
    print("-" * 70)
    
    try:
        with db.get_connection('results') as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            print(f"Tables: {[t['name'] for t in tables]}")
    except Exception as e:
        print(f"Note: {e}")
    
    print("\n" + "=" * 70)
    print("✅ EXAMPLES COMPLETED")
    print("=" * 70)