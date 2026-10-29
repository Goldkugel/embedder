"""
Embedder package.

Exports the Embedder class and its EmbedderConfig Pydantic model, so
consumers can write `from embedder import Embedder, EmbedderConfig`.
"""

from .Embedder       import Embedder
from .EmbedderConfig import EmbedderConfig

__all__ = ["Embedder", "EmbedderConfig"]