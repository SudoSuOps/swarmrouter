"""
SwarmRouter - Intelligent routing for multi-model inference fleets
"""

from .schema import RouterOutput
from .routing_rules import apply_routing_physics

__version__ = "1.0.0"
__all__ = ["RouterOutput", "apply_routing_physics"]
