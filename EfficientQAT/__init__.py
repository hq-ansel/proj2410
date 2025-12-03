"""
EfficientQAT: Efficient Quantization Aware Training for Large Language Models
"""

__version__ = "0.1.0"
__author__ = "EfficientQAT Team"
__email__ = "team@efficientqat.com"

from .core import *
from .quantize import *
from .utils import *

__all__ = [
    "__version__",
    "__author__",
    "__email__",
]