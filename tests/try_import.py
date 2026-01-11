import sys
import os
sys.path.insert(0, os.path.abspath(os.curdir))
try:
    from backend.services.generation.service import ResponseGenerator
    print("Import successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()

