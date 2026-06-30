"""
Controllers package for the modular navigation framework.

This package contains all navigation controller implementations and the base
controller interface that ensures consistency across different planning algorithms.
"""

from .base_controller import BaseController

__all__ = ['BaseController'] 