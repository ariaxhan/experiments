"""●COMPONENT|Ψ:universal_spectroscopy_engine|Ω:orchestrate_sae_analysis_pipeline

UniversalSpectroscopyEngine: Main orchestration class.
Ported from universal-spectroscopy-engine.

This is the primary interface for the Universal Spectroscopy Engine,
coordinating all components (ExcitationController, SAE_Adapter, InterferenceEngine).
"""

from typing import Optional, List
import torch
from transformer_lens import HookedTransformer

from .excitation import ExcitationController
from .sae_adapter import SAE_Adapter, SAELoadError
from .interference import InterferenceEngine
from .spectrum import Spectrum
from .utils import (
    get_device,
    clear_cache,
    model_context,
    cache_model,
    get_cached_model,
    cache_sae,
    get_cached_sae
)


class UniversalSpectroscopyEngine:
    """
    Universal Spectroscopy Engine - Main orchestration class.
    
    Coordinates:
    - ExcitationController (The Slit): Input formatting
    - SAE_Adapter (The Prism): SAE loading and decomposition
    - InterferenceEngine (The Detector): Spectral analysis
    
    Usage:
        >>> engine = UniversalSpectroscopyEngine()
        >>> engine.load_model("gemma-2-2b")
        >>> engine.load_sae("gemma-2-2b", layer=5)
        >>> spectrum = engine.process("Your input text here")
        >>> purity = engine.calculate_purity(spectrum)
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize UniversalSpectroscopyEngine.
        
        Args:
            device: Device to use. If None, auto-detects best device.
        """
        self.device = device if device is not None else get_device()
        self.model: Optional[HookedTransformer] = None
        self.model_name: Optional[str] = None
        
        # Initialize components
        self.excitation_controller = ExcitationController(device=self.device)
        self.sae_adapter = SAE_Adapter(device=self.device)
        self.interference_engine = InterferenceEngine()
    
    def load_model(self, model_name: str, use_cache: bool = True) -> None:
        """
        Load LLM model using transformer_lens.
        
        Args:
            model_name: Name of the model (e.g., "gemma-2-2b")
            use_cache: If True, use cached model if available (default: True)
            
        Raises:
            ValueError: If model cannot be loaded
        """
        # Check cache first
        if use_cache:
            cached_model = get_cached_model(model_name)
            if cached_model is not None:
                print(f"✓ Using cached model: {model_name}")
                self.model = cached_model
                self.model_name = model_name
                return
        
        try:
            print(f"Loading model {model_name} (this may take a while)...")
            self.model = HookedTransformer.from_pretrained(
                model_name,
                device=self.device
            )
            self.model_name = model_name
            
            # Cache the model
            if use_cache:
                cache_model(model_name, self.model)
                print(f"✓ Model cached for future use")
        except Exception as e:
            raise ValueError(
                f"Failed to load model {model_name}. Error: {str(e)}"
            ) from e
    
    def load_sae(
        self,
        model_name: str,
        layer: int,
        release: Optional[str] = None,
        sae_id: Optional[str] = None,
        sae_path: Optional[str] = None,
        width: str = "16k"
    ) -> None:
        """
        Load SAE for specified model and layer.
        
        Args:
            model_name: Name of the model (e.g., "gemma-2-2b")
            layer: Layer number
            release: SAE release name (e.g., "gemma-scope-2b-pt-res-canonical")
            sae_id: SAE identifier (e.g., "layer_5/width_16k/canonical")
            sae_path: Optional path to SAE file. If None, will try to auto-detect.
            width: SAE width (e.g., "16k"). Used if sae_id is None.
            
        Raises:
            SAELoadError: If SAE cannot be loaded
        """
        self.sae_adapter.load_sae(
            model_name=model_name,
            layer=layer,
            release=release,
            sae_id=sae_id,
            width=width
        )
    
    def process(
        self,
        input_text: str,
        layer: Optional[int] = None,
        token_positions: Optional[List[int]] = None
    ) -> Spectrum:
        """
        Process input text and return spectrum.
        
        This is the main processing pipeline:
        1. ExcitationController processes input and extracts activations
        2. SAE_Adapter decomposes activations into feature space
        3. Returns standardized Spectrum object
        
        Args:
            input_text: Input text to process
            layer: Layer number to extract from (uses SAE layer if None)
            token_positions: Specific token positions to extract (None = all tokens)
            
        Returns:
            Spectrum object with feature activations
            
        Raises:
            ValueError: If model or SAE not loaded
        """
        if self.model is None:
            raise ValueError("Model must be loaded before processing. Call load_model() first.")
        
        if not self.sae_adapter.is_loaded():
            raise ValueError("SAE must be loaded before processing. Call load_sae() first.")
        
        # Use SAE layer if layer not specified
        if layer is None:
            layer = self.sae_adapter.layer
            if layer is None:
                raise ValueError("Layer must be specified either in load_sae() or process()")
        
        # Extract activations using ExcitationController
        activations = self.excitation_controller.process(
            input_text=input_text,
            model=self.model,
            layer=layer,
            token_positions=token_positions
        )
        
        # Get metadata
        metadata = self.excitation_controller.get_metadata(
            input_text=input_text,
            layer=layer,
            token_positions=token_positions
        )
        
        # Decompose using SAE_Adapter
        spectrum = self.sae_adapter.decompose(activations, metadata=metadata)
        
        # Normalize spectrum
        spectrum = self.sae_adapter.normalize(spectrum)
        
        return spectrum
    
    def calculate_purity(self, spectrum: Spectrum) -> float:
        """
        Calculate spectral purity (H1: Spectral Purity Hypothesis).
        
        Hallucinations manifest as low spectral purity.
        
        Args:
            spectrum: Input spectrum
            
        Returns:
            Purity score between 0.0 (low purity, possible hallucination) and 1.0 (high purity)
        """
        return self.interference_engine.calculate_purity(spectrum)
    
    def calculate_drift(
        self,
        input_spec: Spectrum,
        output_spec: Spectrum,
        top_k: int = 100
    ) -> float:
        """
        Calculate semantic drift between two spectra (H2: Doppler Shift Hypothesis).
        
        Args:
            input_spec: Input spectrum
            output_spec: Output spectrum
            top_k: Number of top features to compare
            
        Returns:
            Drift magnitude between 0.0 (no drift) and 1.0 (complete drift)
        """
        return self.interference_engine.calculate_drift(
            input_spec, output_spec, top_k=top_k
        )
    
    def detect_absorption(
        self,
        input_spec: Spectrum,
        output_spec: Spectrum,
        threshold: float = 0.5
    ) -> List[int]:
        """
        Detect features present in input but absent in output (H3: Absorption Hypothesis).
        
        Args:
            input_spec: Input spectrum
            output_spec: Output spectrum
            threshold: Minimum intensity threshold
            
        Returns:
            List of absorbed feature indices
        """
        return self.interference_engine.detect_absorption(
            input_spec, output_spec, threshold=threshold
        )
    
    def unload_model(self) -> None:
        """Unload model and clear VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
            self.model_name = None
            clear_cache(self.device)
    
    def unload_sae(self) -> None:
        """Unload SAE and clear VRAM."""
        if self.sae_adapter.is_loaded():
            self.sae_adapter.sae = None
            self.sae_adapter.model_name = None
            self.sae_adapter.layer = None
            clear_cache(self.device)
    
    def cleanup(self) -> None:
        """Clean up all resources."""
        self.unload_model()
        self.unload_sae()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.cleanup()

