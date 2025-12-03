"""Conditional Text Generation Package.

A modern, production-ready package for conditional text generation using
transformer-based language models with comprehensive evaluation metrics.
"""

__version__ = "1.0.0"
__author__ = "AI Projects"
__email__ = "ai@projects.com"

from .models import ConditionalTextGenerator, GPT2ConditionalGenerator
from .data import TextDataset, ConditionalTextDataModule
from .evaluation import TextEvaluator, EvaluationMetrics
from .utils import set_seed, get_device, load_config

__all__ = [
    "ConditionalTextGenerator",
    "GPT2ConditionalGenerator", 
    "TextDataset",
    "ConditionalTextDataModule",
    "TextEvaluator",
    "EvaluationMetrics",
    "set_seed",
    "get_device",
    "load_config",
]
