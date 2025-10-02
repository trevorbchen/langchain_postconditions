# 📚 Postcondition Generation System - Complete Documentation

## 🎯 System Overview

This is a **comprehensive formal verification system** that generates mathematical postconditions for C functions from natural language specifications. It uses **LangChain** for AI orchestration, **Z3 theorem proving** for verification, and **Pydantic** for type-safe data models.

**What it does:**
1. **Input**: Natural language specification (e.g., "Sort an array using bubble sort")
2. **Output**: Formal mathematical postconditions + Z3 verification code + comprehensive analysis

**Example transformation:**
```
Specification → "Sort an array in ascending order"
                    ↓
Pseudocode → void bubble_sort(int* arr, int size)
                    ↓
Postconditions → "∀i,j: 0 ≤ i < j < n → arr[i] ≤ arr[j]"
                    ↓
Z3 Code → from z3 import *; s = Solver(); ...
                    ↓
Validation → ✅ Code valid, constraints satisfiable
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                           │
│  main.py - Interactive CLI / Command-line interface         │
└─────────────────────────────────┬───────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────┐
│              PIPELINE ORCHESTRATOR                          │
│  modules/pipeline/pipeline.py                               │
│  - Coordinates all steps                                    │
│  - Manages async execution                                  │
│  - Aggregates results                                       │
└───────────┬─────────────────────┬───────────────────────────┘
            │                     │
┌───────────▼───────────┐  ┌──────▼──────────────────────────┐
│  PSEUDOCODE GEN       │  │  POSTCONDITION GEN              │
│  modules/pseudocode/  │  │  modules/logic/                 │
│  - C code structure   │  │  - Formal specifications        │
│  - Function sigs      │  │  - Edge case analysis           │
│  - Type inference     │  │  - Quality metrics              │
└───────────┬───────────┘  └──────┬──────────────────────────┘
            │                     │
            │              ┌──────▼──────────────────────────┐
            │              │  Z3 TRANSLATION                 │
            │              │  modules/z3/                    │
            │              │  - Z3 code generation           │
            │              │  - Syntax validation            │
            │              │  - Runtime verification         │
            │              └──────┬──────────────────────────┘
            │                     │
┌───────────▼─────────────────────▼───────────────────────────┐
│                 LANGCHAIN CHAINS                            │
│  core/chains.py                                             │
│  - PseudocodeChain: LLM → C pseudocode                      │
│  - PostconditionChain: LLM → formal logic                   │
│  - Z3TranslationChain: Logic → Z3 code                      │
│  - FormalLogicTranslationChain: Formal → Natural language   │
└───────────┬─────────────────────────────────────────────────┘
            │
┌───────────▼─────────────────────────────────────────────────┐
│              DATA MODELS & STORAGE                          │
│  core/models.py - Pydantic schemas                          │
│  storage/database.py - SQLite persistence                   │
│  storage/vector_store.py - Semantic search (ChromaDB)       │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Directory Structure & File Purposes

### **Root Level**
```
postcondition_system/
├── main.py              # 🎮 Entry point - Interactive CLI interface
├── requirements.txt     # 📦 Python dependencies (LangChain, OpenAI, Z3, etc.)
├── .env                 # 🔐 API keys (OPENAI_API_KEY=sk-...)
├── .gitignore          # 🚫 Git exclusions (.env, __pycache__, etc.)
└── README.md           # 📖 Project documentation
```

### **Core System** (`core/`)
The heart of the system - data models, LangChain chains, and agents.

```
core/
├── __init__.py          # Package initialization
├── models.py            # 🎯 CRITICAL: Pydantic data models
│                        #   - EnhancedPostcondition (15+ fields)
│                        #   - Z3Translation (validation metadata)
│                        #   - Function, FunctionParameter
│                        #   - CompleteEnhancedResult (pipeline output)
│
├── chains.py            # 🔗 LangChain chain implementations
│                        #   - PseudocodeChain: Spec → C code
│                        #   - PostconditionChain: Function → Formal logic
│                        #   - Z3TranslationChain: Logic → Z3 code
│                        #   - FormalLogicTranslationChain: Formal → Natural
│                        #   - ChainFactory: Lazy initialization
│
└── agents.py            # 🤖 LangChain agents (experimental)
                         #   - PostconditionAgent: Autonomous reasoning
                         #   - EdgeCaseAgent: Edge case identification
                         #   - OptimizationAgent: Result refinement
```

**Key Design Decisions:**
- **models.py**: Uses Pydantic 2.0 for type safety, validation, and JSON serialization
- **chains.py**: All chains use `ChatPromptTemplate` for consistent prompting
- **agents.py**: Uses modern LangChain agent patterns (create_tool_calling_agent)

### **Modules** (`modules/`)
Specialized functionality broken into focused modules.

```
modules/
├── __init__.py
│
├── pseudocode/          # 📝 C Pseudocode Generation
│   ├── __init__.py
│   └── pseudocode_generator.py
│       # Generates structured C pseudocode from natural language
│       # Uses PromptsManager for template-based generation
│       # Handles: function signatures, parameters, complexity analysis
│
├── logic/               # 🧮 Formal Postcondition Generation
│   ├── __init__.py
│   └── logic_generator.py
│       # Generates formal mathematical postconditions
│       # Integrates: domain knowledge, edge cases, Z3 theories
│       # Outputs: Enhanced postconditions with 15+ metadata fields
│
├── z3/                  # ✅ Z3 Theorem Prover Integration
│   ├── __init__.py
│   ├── translator.py    # Formal logic → Z3 Python code
│   │                    # Uses Z3CodeValidator for comprehensive validation
│   │                    # Tracks: solver creation, constraints, variables
│   │
│   └── validator.py     # Z3 code validation engine
│       # Pass 1: Syntax validation (AST parsing)
│       # Pass 2: Import validation (Z3 imports present)
│       # Pass 3: Runtime execution (subprocess isolation)
│       # Pass 4: Solver validation (Solver() creation)
│       # Generates: ValidationResult with detailed metrics
│
└── pipeline/            # 🎭 Orchestration Layer
    ├── __init__.py
    └── pipeline.py
        # PostconditionPipeline: End-to-end workflow orchestrator
        # Steps:
        #   1. Generate pseudocode
        #   2. Generate postconditions (per function)
        #   3. Translate to Z3 (batch processing)
        #   4. Compute statistics
        #   5. Generate reports
        # Features: Async execution, error handling, result aggregation
```

**Module Highlights:**
- **pseudocode/**: Stateless generator, uses prompts.yaml templates
- **logic/**: Domain-aware, integrates edge case analysis
- **z3/**: Two-phase (translation + validation), isolated execution
- **pipeline/**: Async-first design, preserves all enriched data

### **Configuration** (`config/`)
All configuration, settings, and prompt templates.

```
config/
├── __init__.py
├── settings.py          # 🔧 Application settings (Pydantic BaseSettings)
│                        #   - OpenAI API configuration
│                        #   - Database paths
│                        #   - Z3 validation settings
│                        #   - LangChain parameters
│                        #   Loads from .env file automatically
│
└── prompts.yaml         # 📋 Comprehensive prompt templates
    # Four major prompt types:
    # 1. pseudocode_generation - C code structure
    # 2. postcondition_generation - Formal logic (MOST IMPORTANT)
    # 3. edge_case_analysis - Edge case identification
    # 4. z3_translation - Z3 code generation
    #
    # Also contains:
    # - domain_knowledge: Patterns for collections, algorithms, etc.
    # - context_building: Templates for context assembly
    # - validation: Quality check prompts
    # - examples: Reference examples (sorting, search, etc.)
```

**Settings Hierarchy:**
1. `.env` file (highest priority)
2. Environment variables
3. Defaults in `settings.py`

**Prompt Template Structure:**
```yaml
postcondition_generation:
  system: "You are an expert..."  # System instructions
  human: "{specification}..."      # User message with variables
  metadata:
    temperature: 0.3
    max_tokens: 2500
```

### **Storage** (`storage/`)
Database and vector store functionality.

```
storage/
├── __init__.py
├── database.py          # 📊 SQLite database manager
│                        #   Databases:
│                        #     - context.db: Domain knowledge, patterns
│                        #     - z3_theories.db: Z3 theory examples
│                        #     - results.db: Pipeline execution results
│                        #   Features:
│                        #     - Context managers for safe connections
│                        #     - Type-safe queries using Pydantic
│                        #     - Automatic result parsing
│
└── vector_store.py      # 🔍 Semantic search (ChromaDB + OpenAI Embeddings)
    # KnowledgeBase class provides:
    #   - Semantic search over domain knowledge
    #   - Edge case pattern matching
    #   - Z3 theory retrieval
    # Uses: LangChain vector store abstractions
    # Persistent storage in: .cache/chroma/
```

**Database Usage:**
```python
from storage.database import DatabaseManager

db = DatabaseManager()
with db.get_connection('context') as conn:
    knowledge = db.query_domain_knowledge('sorting')
```

**Vector Store Usage:**
```python
from storage.vector_store import KnowledgeBase

kb = KnowledgeBase()
kb.initialize()  # Builds from database
results = kb.search("array sorting edge cases", k=5)
```

### **Utilities** (`utils/`)
Helper functions and validation.

```
utils/
├── __init__.py
├── validators.py        # ✔️ Input/output validation
│                        #   - validate_function(): Check Function models
│                        #   - validate_postcondition(): Check postconditions
│                        #   - validate_z3_code(): Syntax validation
│                        #   - validate_specification(): Input validation
│
├── prompt_loader.py     # 📝 Prompt template management
│                        #   PromptsManager class:
│                        #     - Loads prompts.yaml
│                        #     - Variable interpolation
│                        #     - Domain knowledge access
│                        #     - Context building utilities
│
└── batching.py          # ⚡ LLM call optimization
    # Batching utilities for reducing API calls:
    #   - chunk_items(): Split into batches
    #   - batch_process(): Parallel batch processing
    #   - BatchMetrics: Track savings and performance
    # Example: 10 postconditions → 1 API call (90% cost reduction)
```

**Validator Example:**
```python
from utils.validators import validate_function

is_valid, errors = validate_function(func)
if not is_valid:
    print("Errors:", errors)
```

**Batching Example:**
```python
from utils.batching import batch_process

results, metrics = await batch_process(
    items=postconditions,
    process_fn=translate_to_z3,
    batch_size=10
)
print(f"Saved {metrics.saved_llm_calls} API calls!")
```

### **Output** (`output/`)
Generated results and reports.

```
output/
└── pipeline_results/    # 📁 Pipeline execution outputs
    ├── complete_result.json          # Full result (all data)
    ├── validation_report.txt         # Z3 validation summary
    └── pseudocode/
        ├── pseudocode_summary.txt    # Human-readable summary
        └── pseudocode_full.json      # Complete pseudocode structure
```

---

## 🔄 Complete Pipeline Flow

### **Step-by-Step Execution**

```
1. USER INPUT
   ↓
   "Sort an array using bubble sort"
   
2. PIPELINE INITIALIZATION
   ↓
   PostconditionPipeline created
   ChainFactory initialized (lazy loading)
   
3. PSEUDOCODE GENERATION
   ↓
   PseudocodeChain invoked
   LLM generates C function structure
   
4. POSTCONDITION GENERATION (per function)
   ↓
   PostconditionChain invoked with:
     - Function context (signature, parameters)
     - Specification
     - Edge cases
   
5. Z3 TRANSLATION (batch processing)
   ↓
   Z3TranslationChain.atranslate_batch()
   
6. STATISTICS COMPUTATION
   ↓
   Pipeline aggregates metrics
   
7. REPORT GENERATION
   ↓
   Validation report created
   Files saved to output/
   
8. RESULTS RETURNED
   ↓
   CompleteEnhancedResult object with all data
```

---

## 🎯 Key Data Structures

### **EnhancedPostcondition** (Core Data Model)

```python
from core.models import EnhancedPostcondition

postcondition = EnhancedPostcondition(
    # Core fields
    formal_text="∀i,j: 0 ≤ i < j < n → arr[i] ≤ arr[j]",
    natural_language="Array is sorted in ascending order",
    
    # Translation fields
    precise_translation="For every pair of indices i and j...",
    reasoning="This ensures the fundamental sorting property...",
    
    # Edge case analysis
    edge_cases=["empty array", "single element", "duplicates"],
    edge_cases_covered=[
        "Empty array (n=0): vacuously true",
        "Single element (n=1): no pairs to compare"
    ],
    coverage_gaps=["Does not specify stability"],
    
    # Quality metrics (0.0-1.0)
    confidence_score=0.95,
    robustness_score=0.92,
    clarity_score=0.95,
    completeness_score=0.90,
    
    # Z3 integration
    z3_theory="arrays",
    z3_translation=Z3Translation(...)
)
```

### **Z3Translation** (Verification Result)

```python
from core.models import Z3Translation

translation = Z3Translation(
    formal_text="∀i,j: 0 ≤ i < j < n → arr[i] ≤ arr[j]",
    natural_language="Array is sorted",
    z3_code="from z3 import *\n...",
    
    # Validation results
    z3_validation_passed=True,
    z3_validation_status="success",
    
    # Execution metrics
    solver_created=True,
    constraints_added=3,
    variables_declared=4,
    execution_time=0.023
)
```

---

## 🎮 Usage Examples

### **Basic Usage (Interactive)**

```bash
python main.py
```

### **Command-Line Usage**

```bash
# Direct specification
python main.py --spec "Sort an array using bubble sort"

# Custom output directory
python main.py --spec "Binary search" --output results/

# With codebase context
python main.py --spec "Add helper function" --codebase /path/to/code/
```

### **Programmatic Usage**

```python
from modules.pipeline.pipeline import PostconditionPipeline
from pathlib import Path

# Create pipeline
pipeline = PostconditionPipeline(
    codebase_path=Path("path/to/code"),
    validate_z3=True
)

# Process specification (sync)
result = pipeline.process_sync("Sort an array using bubble sort")

# Save results
pipeline.save_results(result, Path("output/"))

# Access results
print(f"Functions: {result.total_functions}")
print(f"Postconditions: {result.total_postconditions}")
print(f"Z3 Success: {result.z3_validation_success_rate:.1%}")

for func_result in result.function_results:
    print(f"\nFunction: {func_result.function_name}")
    for pc in func_result.postconditions:
        print(f"  - {pc.natural_language}")
        print(f"    Quality: {pc.overall_quality_score:.2f}")
```

---

## 🔧 Setup & Installation

### **1. Prerequisites**

- Python 3.8+
- pip package manager
- OpenAI API key

### **2. Installation**

```bash
# Clone repository
git clone <repository-url>
cd postcondition_system

# Install dependencies
pip install -r requirements.txt
```

### **3. Configuration**

Create `.env` file in project root:

```bash
# .env file
OPENAI_API_KEY=sk-your-actual-key-here

# Optional settings
OPENAI_MODEL=gpt-4
TEMPERATURE=0.3
MAX_TOKENS=3000

# Z3 Validation
Z3_VALIDATION_ENABLED=true
Z3_VALIDATION_TIMEOUT_SECONDS=5
```

### **4. Verify Installation**

```bash
# Test settings
python config/settings.py

# Run test
python main.py --spec "Sort an array"
```

---

## 🚀 Quick Start Guide

### **Example 1: Sort Array**

```bash
python main.py
# Enter: "Sort an array using bubble sort"
```

**Expected Output:**
- 1 function generated
- 5-10 postconditions
- Z3 code validated
- Results in `output/pipeline_results/`

### **Example 2: Binary Search**

```bash
python main.py --spec "Binary search for element in sorted array"
```

### **Example 3: Custom Algorithm**

```python
from modules.pipeline.pipeline import PostconditionPipeline

pipeline = PostconditionPipeline()
result = pipeline.process_sync("""
Implement a function that finds the kth smallest element 
in an unsorted array using quickselect algorithm
""")

print(f"Generated {result.total_postconditions} postconditions")
for func_result in result.function_results:
    print(f"\nFunction: {func_result.function_name}")
    for pc in func_result.postconditions:
        print(f"  {pc.formal_text}")
```

---

## 📊 Output Structure

### **complete_result.json**

```json
{
  "session_id": "uuid",
  "specification": "Sort an array",
  "function_results": [
    {
      "function_name": "bubble_sort",
      "postconditions": [
        {
          "formal_text": "∀i,j: 0 ≤ i < j < n → arr[i] ≤ arr[j]",
          "natural_language": "Array is sorted",
          "precise_translation": "For every pair...",
          "reasoning": "This ensures...",
          "confidence_score": 0.95,
          "robustness_score": 0.92,
          "z3_translation": {
            "z3_code": "from z3 import *...",
            "z3_validation_passed": true
          }
        }
      ]
    }
  ],
  "total_postconditions": 5,
  "z3_validation_success_rate": 1.0
}
```

---

## 🐛 Troubleshooting

### **API Key Issues**

```bash
# Check if API key is set
python config/settings.py

# If error, create/update .env file
echo "OPENAI_API_KEY=sk-your-key" > .env
```

### **Import Errors**

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### **Permission Errors (Output)**

```bash
# Create output directory manually
mkdir -p output/pipeline_results
chmod 755 output
```

### **Z3 Validation Failures**

Check `validation_report.txt` for details:
- Syntax errors: Check Z3 code generation prompt
- Runtime errors: Check Z3 installation
- Timeout errors: Increase timeout in settings

---

## 📚 Advanced Topics

### **Custom Domain Knowledge**

Add to `config/prompts.yaml`:

```yaml
domain_knowledge:
  custom_domain:
    description: "Your custom domain"
    common_patterns:
      - "Pattern 1"
      - "Pattern 2"
    edge_cases:
      - "Edge case 1"
```

### **Batch Processing**

```python
from utils.batching import batch_process

# Process multiple specifications
specs = ["Sort array", "Search array", "Reverse array"]
results = [pipeline.process_sync(s) for s in specs]
```

### **Custom Validation Rules**

```python
from utils.validators import validate_postcondition

# Custom validation
def custom_validate(pc):
    is_valid, errors = validate_postcondition(pc)
    
    # Add custom checks
    if pc.confidence_score < 0.8:
        errors.append("Confidence too low")
    
    return len(errors) == 0, errors
```

---

## 🤝 Contributing

See individual module documentation for contribution guidelines.

---

## 📄 License

See LICENSE file for details.

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review module-specific documentation
3. Check output files for detailed error messages