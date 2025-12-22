"""●COMPONENT|Ψ:spectrum_data_structure|Ω:represent_sae_feature_activations

Minimal Spectrum data class for representing LLM feature activations.
Extracted from universal-spectroscopy-engine for self-contained experiments.
"""

from dataclasses import dataclass
from typing import Dict, Any
import torch


@dataclass
class Spectrum:
    """
    Represents a spectral decomposition of LLM activations.
    
    Following the physics metaphor:
    - Wavelengths = Feature indices (which features are active)
    - Intensities = Activation magnitudes (how strongly features activate)
    
    Attributes:
        wavelengths: Feature indices in the SAE dictionary (torch.Tensor[int64])
        intensities: Activation magnitudes (torch.Tensor[float32])
        model_name: Name of the model that generated these activations
        layer: Layer number where activations were extracted
        metadata: Additional metadata (token positions, timestamps, etc.)
    """
    
    wavelengths: torch.Tensor  # Feature indices
    intensities: torch.Tensor   # Activation magnitudes
    model_name: str
    layer: int
    metadata: Dict[str, Any]
    
    def __post_init__(self) -> None:
        """Validate spectrum data and ensure tensors are on CPU."""
        assert len(self.wavelengths) == len(self.intensities), \
            "Wavelengths and intensities must have same length"
        
        # Ensure tensors are on CPU for serialization
        if self.wavelengths.device.type != "cpu":
            self.wavelengths = self.wavelengths.cpu()
        if self.intensities.device.type != "cpu":
            self.intensities = self.intensities.cpu()
    
    def get_top_features(self, k: int = 10) -> torch.Tensor:
        """
        Get top k features by intensity.
        
        Args:
            k: Number of top features to return
            
        Returns:
            Tensor of feature indices (wavelengths) sorted by intensity (descending)
        """
        if k >= len(self.intensities):
            k = len(self.intensities)
        
        _, top_indices = torch.topk(self.intensities, k, largest=True)
        return self.wavelengths[top_indices]
    
    def get_top_intensities(self, k: int = 10) -> torch.Tensor:
        """
        Get top k intensities.
        
        Args:
            k: Number of top intensities to return
            
        Returns:
            Tensor of intensities sorted in descending order
        """
        if k >= len(self.intensities):
            k = len(self.intensities)
        
        top_intensities, _ = torch.topk(self.intensities, k, largest=True)
        return top_intensities
    
    def filter_by_threshold(self, threshold: float) -> "Spectrum":
        """
        Filter spectrum to only include features above threshold.
        
        Args:
            threshold: Minimum intensity threshold
            
        Returns:
            New Spectrum object with filtered features
        """
        mask = self.intensities >= threshold
        return Spectrum(
            wavelengths=self.wavelengths[mask],
            intensities=self.intensities[mask],
            model_name=self.model_name,
            layer=self.layer,
            metadata=self.metadata.copy()
        )
    
    def __len__(self) -> int:
        """Return number of features in spectrum."""
        return len(self.wavelengths)
    
    def __repr__(self) -> str:
        """String representation of Spectrum."""
        return (
            f"Spectrum(model={self.model_name}, layer={self.layer}, "
            f"features={len(self)}, top_intensity={self.intensities.max().item():.4f})"
        )

