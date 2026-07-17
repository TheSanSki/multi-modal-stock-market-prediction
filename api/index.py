import os
import sys

# Compute the base directory path (one up from api/)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
backend_dir = os.path.join(base_dir, "backend")

# Inject paths explicitly to fix relative cross-directory module bindings
for path in [base_dir, backend_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

from backend.main import app
