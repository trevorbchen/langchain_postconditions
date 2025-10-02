#!/usr/bin/env python3
"""
Diagnostic script to test file saving functionality
"""

from pathlib import Path
import sys
import os

print("=" * 80)
print("FILE SAVE DIAGNOSTIC TEST")
print("=" * 80)

# Test 1: Current directory
print("\n1. Current Directory:")
cwd = Path.cwd()
print(f"   {cwd}")
print(f"   Absolute: {cwd.resolve()}")

# Test 2: Create test directory
print("\n2. Creating Test Directory:")
test_dir = cwd / "output" / "test_diagnostic"
try:
    test_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Created: {test_dir}")
    print(f"   Exists: {test_dir.exists()}")
    print(f"   Is directory: {test_dir.is_dir()}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 3: Write test file
print("\n3. Writing Test File:")
test_file = test_dir / "test.txt"
try:
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("Test content\n")
        f.write(f"Current directory: {cwd}\n")
        f.write(f"Test directory: {test_dir}\n")
        f.flush()
        os.fsync(f.fileno())
    
    print(f"   ✅ Wrote: {test_file}")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    sys.exit(1)

# Test 4: Verify file exists
print("\n4. Verifying File:")
if test_file.exists():
    size = test_file.stat().st_size
    content = test_file.read_text()
    print(f"   ✅ File exists")
    print(f"   Size: {size} bytes")
    print(f"   Content preview: {content[:50]}...")
else:
    print(f"   ❌ File does not exist!")
    sys.exit(1)

# Test 5: List directory contents
print("\n5. Directory Contents:")
files = list(test_dir.glob("*"))
print(f"   Found {len(files)} items:")
for f in files:
    print(f"     - {f.name} ({'file' if f.is_file() else 'dir'})")

# Test 6: Test with JSON
print("\n6. Testing JSON File:")
import json
json_file = test_dir / "test.json"
test_data = {
    "test": "data",
    "number": 123,
    "nested": {"key": "value"}
}

try:
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    
    if json_file.exists():
        print(f"   ✅ JSON saved: {json_file}")
        print(f"   Size: {json_file.stat().st_size} bytes")
    else:
        print(f"   ❌ JSON file not found!")
except Exception as e:
    print(f"   ❌ JSON save failed: {e}")

# Test 7: Test pipeline output directory
print("\n7. Testing Pipeline Output Directory:")
pipeline_dir = cwd / "output" / "pipeline_results"
try:
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✅ Created: {pipeline_dir}")
    
    # Try to write a file there
    test_pipeline_file = pipeline_dir / "test_write.txt"
    test_pipeline_file.write_text("Pipeline test")
    
    if test_pipeline_file.exists():
        print(f"   ✅ Can write to pipeline directory")
        test_pipeline_file.unlink()  # Clean up
    else:
        print(f"   ❌ Cannot write to pipeline directory")
        
except Exception as e:
    print(f"   ❌ Pipeline directory test failed: {e}")

# Test 8: Check permissions
print("\n8. Checking Permissions:")
try:
    import stat
    mode = test_dir.stat().st_mode
    print(f"   Directory mode: {oct(mode)}")
    print(f"   Readable: {os.access(test_dir, os.R_OK)}")
    print(f"   Writable: {os.access(test_dir, os.W_OK)}")
    print(f"   Executable: {os.access(test_dir, os.X_OK)}")
except Exception as e:
    print(f"   ❌ Permission check failed: {e}")

# Final summary
print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)
print(f"\nTest files created in: {test_dir}")
print(f"You can verify manually by checking: {test_dir.absolute()}")
print("\nIf all tests passed, file saving should work correctly.")
print("=" * 80)