"""
Fruit Classifier MLOps Package
"""

__version__ = "1.0.0"
__author__ = "ThierrySHYAKA"

from . import preprocessing
from . import model
from . import prediction

__all__ = ['preprocessing', 'model', 'prediction']