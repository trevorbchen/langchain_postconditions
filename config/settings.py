"""
Configuration settings loaded from environment variables.
Place your API key in a .env file in the root directory:

.env file should contain:
OPENAI_API_KEY=sk-...your-key-here...
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional
import os


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    
    Example .env file:
        OPENAI_API_KEY=sk-proj-...
        OPENAI_MODEL=gpt-4
        TEMPERATURE=0.3
    """
    
    # ============================================================================
    # OpenAI Configuration
    # ============================================================================
    openai_api_key: str  # Required - will fail if not found
    openai_model: str = "gpt-4"
    temperature: float = 0.3
    max_tokens: int = 3000
    
    # ============================================================================
    # Database Paths
    # ============================================================================
    context_db: Path = Path("context.db")
    z3_theories_db: Path = Path("z3_theories.db")
    results_db: Path = Path("results.db")
    
    # ============================================================================
    # LangChain Settings
    # ============================================================================
    chunk_size: int = 1000
    chunk_overlap: int = 200
    enable_streaming: bool = True
    
    # ============================================================================
    # Caching Configuration
    # ============================================================================
    enable_cache: bool = True
    cache_dir: Path = Path(".cache")
    llm_cache_db: Path = Path(".cache/llm_cache.db")
    
    # ============================================================================
    # Vector Store Settings
    # ============================================================================
    vector_store_dir: Path = Path(".cache/chroma")
    embedding_model: str = "text-embedding-3-small"
    
    # ============================================================================
    # Agent Configuration
    # ============================================================================
    max_retries: int = 3
    request_timeout: int = 120
    verbose: bool = True
    
    # ============================================================================
    # Pipeline Settings
    # ============================================================================
    max_parallel_tasks: int = 5
    enable_monitoring: bool = False
    langsmith_api_key: Optional[str] = None
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",  # Look in project root
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"  # Ignore extra fields in .env
    )
    
    def __init__(self, **kwargs):
        """Initialize settings and create necessary directories."""
        super().__init__(**kwargs)
        
        # Create cache directories if they don't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.llm_cache_db.parent.mkdir(parents=True, exist_ok=True)
    
    @property
    def openai_client_kwargs(self) -> dict:
        """Returns kwargs for initializing OpenAI client."""
        return {
            "api_key": self.openai_api_key,
            "max_retries": self.max_retries,
            "timeout": self.request_timeout
        }
    
    @property
    def llm_kwargs(self) -> dict:
        """Returns kwargs for LangChain ChatOpenAI initialization."""
        return {
            "model": self.openai_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "openai_api_key": self.openai_api_key,
            "streaming": self.enable_streaming,
            "max_retries": self.max_retries,
            "request_timeout": self.request_timeout
        }


# ============================================================================
# Global Settings Instance
# ============================================================================
settings = Settings()


# ============================================================================
# Helper Functions
# ============================================================================

def validate_api_key() -> bool:
    """
    Validates that the OpenAI API key is properly set.
    
    Returns:
        bool: True if API key is valid format, False otherwise
    """
    api_key = settings.openai_api_key
    
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found in environment!")
        print("Please create a .env file with: OPENAI_API_KEY=sk-...")
        return False
    
    if not api_key.startswith("sk-"):
        print("⚠️  WARNING: API key doesn't start with 'sk-'. This might be invalid.")
        return False
    
    print(f"✅ API key loaded: {api_key[:20]}...{api_key[-4:]}")
    return True


def print_settings():
    """Print current settings (hiding sensitive info)."""
    print("\n" + "="*70)
    print("CURRENT SETTINGS")
    print("="*70)
    print(f"OpenAI Model:        {settings.openai_model}")
    print(f"Temperature:         {settings.temperature}")
    print(f"Max Tokens:          {settings.max_tokens}")
    print(f"Cache Enabled:       {settings.enable_cache}")
    print(f"Streaming Enabled:   {settings.enable_streaming}")
    print(f"Context DB:          {settings.context_db}")
    print(f"Z3 Theories DB:      {settings.z3_theories_db}")
    print(f"Vector Store:        {settings.vector_store_dir}")
    print(f"API Key:             {settings.openai_api_key[:20]}...{settings.openai_api_key[-4:]}")
    print("="*70 + "\n")


# ============================================================================
# Validate on import
# ============================================================================
if __name__ != "__main__":
    validate_api_key()


# ============================================================================
# Test script
# ============================================================================
if __name__ == "__main__":
    print("Testing settings configuration...")
    print_settings()
    
    if validate_api_key():
        print("\n✅ All settings loaded successfully!")
        
        # Test OpenAI connection
        print("\nTesting OpenAI connection...")
        try:
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(**settings.llm_kwargs)
            response = llm.invoke("Say 'Settings working!'")
            print(f"✅ OpenAI Response: {response.content}")
            
        except Exception as e:
            print(f"❌ OpenAI connection failed: {e}")
    else:
        print("\n❌ Settings validation failed!")
        print("\nTo fix:")
        print("1. Create a .env file in your project root")
        print("2. Add: OPENAI_API_KEY=sk-your-actual-key-here")
        print("3. Run this script again")