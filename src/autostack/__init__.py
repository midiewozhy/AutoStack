"""
AutoStack: Automated Stacking Ensemble Library
"""

from .estimators.auto_stack import AutoStackRegressor
from .registry import BASE_LEARNER_REGISTRY, META_LEARNER_REGISTRY

__all__ = [
    "AutoStackRegressor",
    "BASE_LEARNER_REGISTRY",
    "META_LEARNER_REGISTRY",
]

__version__ = "0.1.0"