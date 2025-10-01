#!/usr/bin/env python3
"""
Debug script for pseudocode generation issues.

This will help identify what's failing in the pseudocode chain.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.chains import PseudocodeChain
import logging

# Enable detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_pseudocode_generation():
    """Test pseudocode generation step by step."""
    
    print("=" * 70)
    print("PSEUDOCODE GENERATION DEBUG")
    print("=" * 70)
    
    # Step 1: Check API key
    print("\n1. Checking API key...")
    try:
        from config.settings import settings
        api_key = settings.openai_api_key
        
        if not api_key:
            print("❌ ERROR: OPENAI_API_KEY not found in environment!")
            print("\nFix:")
            print("  1. Create .env file in project root")
            print("  2. Add: OPENAI_API_KEY=sk-your-key-here")
            return False
        
        if not api_key.startswith("sk-"):
            print(f"❌ WARNING: API key doesn't start with 'sk-': {api_key[:10]}...")
            return False
        
        print(f"✅ API key found: {api_key[:20]}...{api_key[-4:]}")
        
    except Exception as e:
        print(f"❌ ERROR loading settings: {e}")
        return False
    
    # Step 2: Create pseudocode chain
    print("\n2. Creating PseudocodeChain...")
    try:
        chain = PseudocodeChain()
        print("✅ PseudocodeChain created successfully")
    except Exception as e:
        print(f"❌ ERROR creating chain: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Test simple generation
    print("\n3. Testing simple pseudocode generation...")
    print("   Specification: 'Sort an array'")
    
    try:
        result = chain.generate("Sort an array")
        
        print("\n4. Checking result...")
        print(f"   Type: {type(result)}")
        print(f"   Has functions: {hasattr(result, 'functions')}")
        
        if hasattr(result, 'functions'):
            print(f"   Number of functions: {len(result.functions)}")
            
            if result.functions:
                print("\n✅ SUCCESS! Pseudocode generated:")
                for func in result.functions:
                    print(f"      - {func.name}: {func.description[:50]}...")
                return True
            else:
                print("❌ ERROR: No functions in result")
                print(f"   Result: {result}")
                return False
        else:
            print("❌ ERROR: Result doesn't have 'functions' attribute")
            print(f"   Result: {result}")
            return False
        
    except Exception as e:
        print(f"\n❌ ERROR during generation: {type(e).__name__}")
        print(f"   Message: {str(e)}")
        
        # Detailed error info
        if "timeout" in str(e).lower():
            print("\n💡 TIMEOUT ERROR:")
            print("   - Request took longer than 180 seconds")
            print("   - Try: Use faster model (gpt-4-turbo-preview)")
            print("   - Try: Simplify the specification")
        
        elif "rate" in str(e).lower():
            print("\n💡 RATE LIMIT ERROR:")
            print("   - You've exceeded OpenAI API rate limits")
            print("   - Wait a few minutes and try again")
            print("   - Check your OpenAI usage dashboard")
        
        elif "api" in str(e).lower() or "key" in str(e).lower():
            print("\n💡 API KEY ERROR:")
            print("   - Check your API key is correct")
            print("   - Verify key is active in OpenAI dashboard")
            print("   - Make sure key has proper permissions")
        
        else:
            print("\n💡 UNKNOWN ERROR:")
            print("   Full traceback:")
            import traceback
            traceback.print_exc()
        
        return False


def test_direct_openai():
    """Test direct OpenAI API call to isolate the issue."""
    
    print("\n" + "=" * 70)
    print("DIRECT OPENAI API TEST")
    print("=" * 70)
    
    try:
        from config.settings import settings
        import openai
        
        print("\n1. Creating OpenAI client...")
        client = openai.OpenAI(api_key=settings.openai_api_key)
        print("✅ Client created")
        
        print("\n2. Sending test request...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say 'API is working!'"}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        print("✅ Response received:")
        print(f"   {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ Direct API call failed: {type(e).__name__}")
        print(f"   {str(e)}")
        
        if "invalid" in str(e).lower() or "authentication" in str(e).lower():
            print("\n💡 API KEY IS INVALID")
            print("   1. Check your .env file")
            print("   2. Verify key on OpenAI dashboard")
            print("   3. Generate a new key if needed")
        
        return False


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           PSEUDOCODE GENERATION DEBUG SCRIPT                      ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")
    
    # Run tests
    success1 = test_direct_openai()
    
    if success1:
        print("\n" + "="*70)
        success2 = test_pseudocode_generation()
        
        if success2:
            print("\n" + "="*70)
            print("✅ ALL TESTS PASSED!")
            print("="*70)
            print("\nYour pseudocode generation is working correctly.")
            print("The issue must be elsewhere in the pipeline.")
        else:
            print("\n" + "="*70)
            print("❌ PSEUDOCODE GENERATION FAILED")
            print("="*70)
            print("\nDirect API works, but pseudocode chain fails.")
            print("Check the error messages above for details.")
    else:
        print("\n" + "="*70)
        print("❌ API CONNECTION FAILED")
        print("="*70)
        print("\nFix your API key first, then run this script again.")
    
    sys.exit(0 if success1 and success2 else 1)