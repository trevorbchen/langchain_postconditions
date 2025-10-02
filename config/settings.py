"""
Configuration settings loaded from environment variables.
Place your API key in a .env file in the root directory:

.env file should contain:
OPENAI_API_KEY=sk-...your-key-here...
"""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional
import os


# ============================================================================
# Z3 VALIDATION CONFIGURATION (NEW)
# ============================================================================

class Z3ValidationConfig(BaseModel):
    """
    Configuration for Z3 code validation.
    
    Controls how Z3 code is validated during translation.
    All settings can be overridden via environment variables with Z3_VALIDATION_ prefix.
    
    Example .env entries:
        Z3_VALIDATION_ENABLED=true
        Z3_VALIDATION_TIMEOUT_SECONDS=10
        Z3_VALIDATION_EXECUTION_METHOD=subprocess
    """
    
    # Enable/disable validation
    enabled: bool = Field(
        default=True,
        description="Enable Z3 code validation"
    )
    
    # Timeout settings
    timeout_seconds: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Maximum execution time for Z3 code (seconds)"
    )
    
    # Validation passes to enable
    validate_syntax: bool = Field(
        default=True,
        description="Enable Python syntax validation (AST parsing)"
    )
    
    validate_imports: bool = Field(
        default=True,
        description="Check that Z3 imports are present"
    )
    
    validate_execution: bool = Field(
        default=True,
        description="Execute Z3 code to check for runtime errors"
    )
    
    validate_solver: bool = Field(
        default=True,
        description="Verify that Solver() is created and used"
    )
    
    # Execution method
    execution_method: str = Field(
        default="subprocess",
        description="Execution method: 'subprocess' (safer) or 'exec' (faster)"
    )
    
    # Performance thresholds
    max_execution_time: float = Field(
        default=10.0,
        description="Warn if execution takes longer than this (seconds)"
    )
    
    # Quality thresholds
    min_solver_creation_rate: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable rate of solver creation (0.0-1.0)"
    )
    
    min_success_rate: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable validation success rate (0.0-1.0)"
    )
    
    # Reporting
    generate_reports: bool = Field(
        default=True,
        description="Generate validation reports after pipeline runs"
    )
    
    verbose_errors: bool = Field(
        default=True,
        description="Include detailed error messages in validation results"
    )
    
    class Config:
        env_prefix = "Z3_VALIDATION_"
        extra = "ignore"


# ============================================================================
# MAIN SETTINGS
# ============================================================================

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    
    Example .env file:
        OPENAI_API_KEY=sk-proj-...
        OPENAI_MODEL=gpt-4
        TEMPERATURE=0.3
        Z3_VALIDATION_ENABLED=true
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
    
    # ============================================================================
    # Z3 VALIDATION SETTINGS (NEW)
    # ============================================================================
    z3_validation: Z3ValidationConfig = Field(
        default_factory=Z3ValidationConfig,
        description="Z3 code validation configuration"
    )
    
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
    
    # 🆕 Z3 Validation Settings
    print("\n" + "-"*70)
    print("Z3 VALIDATION SETTINGS (NEW)")
    print("-"*70)
    print(f"Validation Enabled:  {settings.z3_validation.enabled}")
    print(f"Timeout:             {settings.z3_validation.timeout_seconds}s")
    print(f"Execution Method:    {settings.z3_validation.execution_method}")
    print(f"Validate Syntax:     {settings.z3_validation.validate_syntax}")
    print(f"Validate Imports:    {settings.z3_validation.validate_imports}")
    print(f"Validate Execution:  {settings.z3_validation.validate_execution}")
    print(f"Validate Solver:     {settings.z3_validation.validate_solver}")
    print(f"Generate Reports:    {settings.z3_validation.generate_reports}")
    
    print("="*70 + "\n")


def print_z3_validation_settings():
    """Print detailed Z3 validation settings."""
    print("\n" + "="*70)
    print("Z3 VALIDATION CONFIGURATION")
    print("="*70)
    
    config = settings.z3_validation
    
    print("\n📋 Basic Settings:")
    print(f"  Enabled:             {config.enabled}")
    print(f"  Timeout:             {config.timeout_seconds}s")
    print(f"  Execution Method:    {config.execution_method}")
    
    print("\n✅ Validation Passes:")
    print(f"  Syntax Check:        {config.validate_syntax}")
    print(f"  Import Check:        {config.validate_imports}")
    print(f"  Execution Check:     {config.validate_execution}")
    print(f"  Solver Check:        {config.validate_solver}")
    
    print("\n📊 Quality Thresholds:")
    print(f"  Min Success Rate:    {config.min_success_rate:.1%}")
    print(f"  Min Solver Creation: {config.min_solver_creation_rate:.1%}")
    print(f"  Max Execution Time:  {config.max_execution_time}s")
    
    print("\n📄 Reporting:")
    print(f"  Generate Reports:    {config.generate_reports}")
    print(f"  Verbose Errors:      {config.verbose_errors}")
    
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
        
        # Print Z3 validation settings
        print_z3_validation_settings()
        
        # Test OpenAI connection
        print("\nTesting OpenAI connection...")
        try:
            from langchain_openai import ChatOpenAI
            
            llm = ChatOpenAI(**settings.llm_kwargs)
            response = llm.invoke("Say 'Settings working!'")
            print(f"✅ OpenAI Response: {response.content}")
            
        except Exception as e:
            print(f"❌ OpenAI connection failed: {e}")
            
        # Test Z3 validation settings access
        print("\nTesting Z3 validation settings access...")
        try:
            print(f"  Timeout: {settings.z3_validation.timeout_seconds}s")
            print(f"  Method: {settings.z3_validation.execution_method}")
            print(f"  Enabled: {settings.z3_validation.enabled}")
            print("✅ Z3 validation settings accessible!")
        except Exception as e:
            print(f"❌ Z3 validation settings error: {e}")
            
    else:
        print("\n❌ Settings validation failed!")
        print("\nTo fix:")
        print("1. Create a .env file in your project root")
        print("2. Add: OPENAI_API_KEY=sk-your-actual-key-here")
        print("3. Optionally add Z3 validation settings:")
        print("   Z3_VALIDATION_ENABLED=true")
        print("   Z3_VALIDATION_TIMEOUT_SECONDS=10")
        print("   Z3_VALIDATION_EXECUTION_METHOD=subprocess")
        print("4. Run this script again")