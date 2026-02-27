#!/usr/bin/env python3
import sys
import os

print("=== SIMPLE TEST ===", flush=True)
print(f"Python: {sys.version}", flush=True)
print(f"Env vars: {list(os.environ.keys())}", flush=True)
print("Test completed successfully", flush=True)
