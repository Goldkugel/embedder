"""
Embedder package.

Exports the Embedder class and its EmbedderConfig Pydantic model for use
by the adapter framework and any other consumers of this package.
"""

from .Embedder       import Embedder
from .EmbedderConfig import EmbedderConfig

__all__ = ["Embedder", "EmbedderConfig"]