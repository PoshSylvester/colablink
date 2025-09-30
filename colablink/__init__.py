"""
ColabLink - Connect your local IDE to Google Colab GPU runtime.

Work locally with all your files and terminal, while executing code on Colab's GPU.
"""

from .runtime import ColabRuntime
from .client import LocalClient

__version__ = "1.0.0"
__all__ = ["ColabRuntime", "LocalClient"]

