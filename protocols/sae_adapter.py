"""●COMPONENT|Ψ:sae_adapter|Ω:sae_loading_activation_decomposition

SAE_Adapter: The Prism - Loads SAEs and normalizes outputs to Spectrum format.
Ported from universal-spectroscopy-engine.
"""

from typing import Optional, Dict, Any
import torch

from .spectrum import Spectrum
from .utils import (
    get_device,
    verify_device_compatibility,
    cache_sae,
    get_cached_sae
)

# Import sae_lens library
try:
    from sae_lens import SAE
    SAE_LENS_AVAILABLE = True
except ImportError:
    SAE_LENS_AVAILABLE = False
    SAE = None


class SAELoadError(Exception):
    """Raised when SAE fails to load."""
    pass


class SAE_Adapter:
    """
    The Prism: Loads SAEs and normalizes outputs into Spectrum format.
    
    This component:
    - Loads SAEs using sae_lens library (Gemma-Scope, etc.)
    - Decomposes activations into feature space
    - Normalizes outputs into standardized Spectrum objects
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize SAE_Adapter.
        
        Args:
            device: Device to use. If None, auto-detects best device.
        """
        self.device = device if device is not None else get_device()
        self.sae: Optional[Any] = None  # SAE from sae_lens
        self.model_name: Optional[str] = None
        self.layer: Optional[int] = None
        self.sae_release: Optional[str] = None
        self.sae_id: Optional[str] = None
        self.sae_type: str = "sae_lens"  # Default SAE type
        self.hook_name: Optional[str] = None
    
    def load_sae_from_pretrained(
        self,
        release: str,
        sae_id: str,
        model_name: Optional[str] = None
    ) -> None:
        """
        Load SAE from sae_lens pretrained releases (e.g., Gemma-Scope).
        
        Args:
            release: SAE release name (e.g., "gemma-scope-2b-pt-res-canonical")
            sae_id: SAE identifier (e.g., "layer_5/width_16k/canonical")
            model_name: Name of the model (for metadata)
            
        Raises:
            SAELoadError: If SAE cannot be loaded
        """
        if not SAE_LENS_AVAILABLE:
            raise SAELoadError(
                "sae_lens library not available. "
                "Install with: pip install sae-lens"
            )
        
        try:
            # Load SAE using sae_lens
            self.sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=str(self.device)
            )
            
            self.sae_release = release
            self.sae_id = sae_id
            self.model_name = model_name or "unknown"
            
            # Extract layer from sae_id (e.g., "layer_5/width_16k/canonical" -> 5)
            if "layer_" in sae_id:
                layer_str = sae_id.split("layer_")[1].split("/")[0]
                self.layer = int(layer_str)
            else:
                self.layer = -1
            
            # Set hook name based on layer
            if self.layer >= 0:
                self.hook_name = f"blocks.{self.layer}.hook_resid_post"
            
        except Exception as e:
            raise SAELoadError(
                f"Failed to load SAE from {release}/{sae_id}. Error: {str(e)}"
            ) from e
    
    def load_sae(
        self,
        model_name: str,
        layer: int,
        release: Optional[str] = None,
        sae_id: Optional[str] = None,
        width: str = "16k",
        use_cache: bool = True
    ) -> None:
        """
        Load SAE for specified model and layer using sae_lens.
        
        This method loads SAEs from Gemma-Scope or other sae_lens releases.
        
        Args:
            model_name: Name of the model (e.g., "gemma-2-2b")
            layer: Layer number
            release: SAE release name. If None, infers from model_name
            sae_id: SAE identifier. If None, constructs from layer and width
            width: SAE width (default: "16k")
            use_cache: If True, use cached SAE if available (default: True)
            
        Raises:
            SAELoadError: If SAE cannot be loaded
        """
        if not SAE_LENS_AVAILABLE:
            raise SAELoadError(
                "sae_lens library not available. "
                "Install with: pip install sae-lens"
            )
        
        # Auto-detect release from model name if not provided
        if release is None:
            if "gemma-2-2b" in model_name.lower() or "gemma2" in model_name.lower():
                release = "gemma-scope-2b-pt-res-canonical"
            else:
                raise SAELoadError(
                    f"Cannot auto-detect SAE release for model '{model_name}'. "
                    f"Please provide 'release' parameter explicitly.\n"
                    f"Example releases: 'gemma-scope-2b-pt-res-canonical'"
                )
        
        # Construct sae_id if not provided
        if sae_id is None:
            sae_id = f"layer_{layer}/width_{width}/canonical"
        
        # Check cache first
        sae_cache_key = f"{release}::{sae_id}"
        if use_cache:
            cached_sae = get_cached_sae(sae_cache_key)
            if cached_sae is not None:
                print(f"✓ Using cached SAE: {sae_id}")
                self.sae = cached_sae
                self.sae_release = release
                self.sae_id = sae_id
                self.model_name = model_name
                self.layer = layer
                if self.layer >= 0:
                    self.hook_name = f"blocks.{self.layer}.hook_resid_post"
                return
        
        # Load using sae_lens
        print(f"Loading SAE {sae_id} (this may take a while)...")
        self.load_sae_from_pretrained(
            release=release,
            sae_id=sae_id,
            model_name=model_name
        )
        
        # Cache the SAE
        if use_cache and self.sae is not None:
            cache_sae(sae_cache_key, self.sae)
            print(f"✓ SAE cached for future use")
    
    def _encode_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Encode activations using the loaded SAE (sae_lens).
        
        Args:
            activations: Input activations tensor [batch*tokens, d_model]
            
        Returns:
            Feature activations tensor [batch*tokens, d_sae]
        """
        if self.sae is None:
            raise ValueError("SAE must be loaded before encoding.")
        
        with torch.no_grad():
            # sae_lens SAE has an encode() method
            feature_activations = self.sae.encode(activations)
            return feature_activations
    
    def decompose(
        self,
        activations: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Spectrum:
        """
        Decompose activations into feature space and create Spectrum.
        
        Args:
            activations: Activations tensor from model (shape: [batch, tokens, hidden_dim])
            metadata: Additional metadata to include in spectrum
            
        Returns:
            Spectrum object with wavelengths (feature indices) and intensities
            
        Raises:
            ValueError: If SAE is not loaded
        """
        if self.sae is None:
            raise ValueError("SAE must be loaded before decomposition. Call load_sae() first.")
        
        # Ensure activations are on correct device
        activations = activations.to(self.device)
        
        # Flatten batch and token dimensions for processing
        original_shape = activations.shape
        activations_flat = activations.view(-1, activations.shape[-1])
        
        # Run SAE forward pass to get feature activations
        try:
            feature_activations = self._encode_activations(activations_flat)
        except Exception as e:
            raise ValueError(
                f"Failed to encode activations with SAE. Error: {str(e)}"
            ) from e
        
        # Get non-zero features (sparse representation)
        # Find features with non-zero activations
        feature_mask = feature_activations.abs() > 1e-6  # Threshold for "active"
        
        # Get indices and values of active features
        active_indices = torch.nonzero(feature_mask, as_tuple=False)
        
        if len(active_indices) == 0:
            # No active features - return empty spectrum
            wavelengths = torch.tensor([], dtype=torch.int64, device=feature_activations.device)
            intensities = torch.tensor([], dtype=torch.float32, device=feature_activations.device)
        else:
            # Extract feature indices (wavelengths) and intensities
            feature_idx = active_indices[:, 1]  # Feature index
            batch_token_idx = active_indices[:, 0]  # Batch*token index
            
            # Get intensities for active features
            intensities_values = feature_activations[batch_token_idx, feature_idx]
            
            # Aggregate by feature index (sum intensities across tokens)
            # This gives us the total activation per feature
            unique_features, inverse_indices = torch.unique(
                feature_idx, return_inverse=True
            )
            aggregated_intensities = torch.zeros(
                len(unique_features), 
                dtype=torch.float32,
                device=feature_activations.device
            )
            aggregated_intensities.scatter_add_(
                0, inverse_indices, intensities_values.abs()
            )
            
            wavelengths = unique_features
            intensities = aggregated_intensities
        
        # Combine metadata
        combined_metadata = {
            "original_shape": list(original_shape),
            "sae_type": self.sae_type,
            **(metadata or {})
        }
        
        return Spectrum(
            wavelengths=wavelengths,
            intensities=intensities,
            model_name=self.model_name or "unknown",
            layer=self.layer or -1,
            metadata=combined_metadata
        )
    
    def normalize(self, spectrum: Spectrum) -> Spectrum:
        """
        Normalize spectrum to standard format.
        
        This ensures:
        - Wavelengths are sorted
        - Intensities are normalized (optional)
        - Metadata is standardized
        
        Args:
            spectrum: Input spectrum
            
        Returns:
            Normalized spectrum
        """
        # Sort by intensity (descending)
        sorted_indices = torch.argsort(spectrum.intensities, descending=True)
        
        return Spectrum(
            wavelengths=spectrum.wavelengths[sorted_indices],
            intensities=spectrum.intensities[sorted_indices],
            model_name=spectrum.model_name,
            layer=spectrum.layer,
            metadata=spectrum.metadata.copy()
        )
    
    def is_loaded(self) -> bool:
        """Check if SAE is loaded."""
        return self.sae is not None



