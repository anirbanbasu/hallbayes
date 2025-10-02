"""HallBayes: EDFL/B2T/ISR hallucination risk calculator for LLMs"""

from .hallucination_toolkit import (
    OpenAIBackend,
    OpenAIItem,
    OpenAIPlanner,
    ItemMetrics,
    AggregateReport,
    SLACertificate,
    Decision,
    make_sla_certificate,
    save_sla_certificate_json,
    generate_answer_if_allowed,
    # Core math functions
    kl_bernoulli,
    bits_to_trust,
    roh_upper_bound,
    isr,
    delta_bar_from_probs,
    decision_rule,
)

# Import backends with graceful fallback for optional dependencies
try:
    from .htk_backends import AnthropicBackend
except ImportError:
    AnthropicBackend = None

try:
    from .htk_backends import HuggingFaceBackend
except ImportError:
    HuggingFaceBackend = None

try:
    from .htk_backends import OllamaBackend
except ImportError:
    OllamaBackend = None

try:
    from .htk_backends import OpenRouterBackend
except ImportError:
    OpenRouterBackend = None

__version__ = "0.1.0"
__author__ = "Hassana Labs"

# Export all available components
__all__ = [
    "OpenAIBackend",
    "OpenAIItem",
    "OpenAIPlanner",
    "ItemMetrics",
    "AggregateReport",
    "SLACertificate",
    "Decision",
    "make_sla_certificate",
    "save_sla_certificate_json",
    "generate_answer_if_allowed",
    "AnthropicBackend",
    "HuggingFaceBackend",
    "OllamaBackend",
    "OpenRouterBackend",
]