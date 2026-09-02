"""
ChromaDB compatibility module.
Ensures UTF-8 console output on Windows and graceful fallback for optional gRPC telemetry.
"""
import os
import sys
import warnings
from unittest.mock import MagicMock

# Suppress harmless OpenTelemetry / gRPC version warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="opentelemetry")
warnings.filterwarnings("ignore", message=".*grpc.*")

# Configure console streams to use UTF-8 on Windows to avoid UnicodeEncodeError
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

def setup_chroma_compat():
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    os.environ["CHROMA_TELEMETRY"] = "False"
    try:
        import grpc
    except Exception:
        if "grpc" not in sys.modules or isinstance(sys.modules.get("grpc"), (type(None), MagicMock)):
            g = MagicMock()
            g.__version__ = "1.65.0"
            sys.modules["grpc"] = g
            sys.modules["grpc._compression"] = MagicMock()
            sys.modules["grpc._cython"] = MagicMock()

setup_chroma_compat()
