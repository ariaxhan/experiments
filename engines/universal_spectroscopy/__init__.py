"""●COMPONENT|Ψ:universal_spectroscopy_engine|Ω:sae_based_feature_analysis

Universal Spectroscopy Engine - SAE-based feature analysis.
Ported from universal-spectroscopy-engine repository.

This engine treats LLM activations as light spectra, using SAEs to decompose
activations into interpretable features and analyze their properties.

Main Components:
- UniversalSpectroscopyEngine: Main orchestrator
- ExcitationController: Input processing and activation extraction
- SAE_Adapter: SAE loading and feature decomposition
- InterferenceEngine: Spectral analysis (purity, drift, absorption)
- Spectrum: Data structure for feature activations
"""

from .engine import UniversalSpectroscopyEngine
from .excitation import ExcitationController
from .sae_adapter import SAE_Adapter, SAELoadError
from .interference import InterferenceEngine
from .spectrum import Spectrum
from .utils import get_device, clear_cache, model_context

__all__ = [
    "UniversalSpectroscopyEngine",
    "ExcitationController",
    "SAE_Adapter",
    "SAELoadError",
    "InterferenceEngine",
    "Spectrum",
    "get_device",
    "clear_cache",
    "model_context",
]

