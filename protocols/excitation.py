"""●COMPONENT|Ψ:excitation_controller|Ω:input_processing_activation_extraction

ExcitationController: The Slit - Manages input formatting and feature steering.
Ported from universal-spectroscopy-engine.
"""

from typing import List, Optional, Dict, Any
import torch
from transformer_lens import HookedTransformer

from .utils import get_device


class ExcitationController:
    """
    The Slit: Manages input formatting and feature steering.
    
    Capabilities:
    - Standard input processing: Format text for model processing
    - Activation extraction: Get layer activations from transformer models
    - Monochromatic steering: Force specific features to activate
    - Pulse trains: Inject noise for robustness testing
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize ExcitationController.
        
        Args:
            device: Device to use. If None, auto-detects best device.
        """
        self.device = device if device is not None else get_device()
    
    def process(
        self,
        input_text: str,
        model: HookedTransformer,
        layer: int,
        token_positions: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Process input text and extract activations from specified layer.
        
        Args:
            input_text: Input text to process
            model: HookedTransformer model instance
            layer: Layer number to extract activations from
            token_positions: Specific token positions to extract (None = all tokens)
            
        Returns:
            Tensor of activations from the specified layer
        """
        # Tokenize input
        tokens = model.to_tokens(input_text)
        tokens = tokens.to(self.device)
        
        # Run model and extract activations
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, return_type=None)
            
            # Get activations from specified layer
            activation_key = f"blocks.{layer}.hook_resid_post"
            activations = cache[activation_key]
            
            # Select token positions if specified
            if token_positions is not None:
                activations = activations[:, token_positions, :]
            
            # Return activations (shape: [batch, tokens, hidden_dim])
            return activations
    
    def monochromatic_steering(
        self,
        activations: torch.Tensor,
        feature_indices: List[int],
        intensity: float = 1.0
    ) -> torch.Tensor:
        """
        Force specific features to activate (monochromatic steering).
        
        This simulates "forcing" certain concepts to be present in the spectrum.
        
        Args:
            activations: Base activations tensor
            feature_indices: List of feature indices to force
            intensity: Intensity multiplier for forced features
            
        Returns:
            Modified activations tensor
        """
        modified = activations.clone()
        
        # Note: This is a placeholder - actual feature steering would require
        # knowledge of the SAE feature dictionary. This would be implemented
        # after SAE is loaded and we know the feature space.
        # For now, we'll add a marker in metadata that steering was applied.
        
        return modified
    
    def pulse_train(
        self,
        activations: torch.Tensor,
        noise_level: float = 0.1,
        noise_type: str = "gaussian"
    ) -> torch.Tensor:
        """
        Inject noise into activations (pulse train).
        
        Useful for robustness testing and detecting sensitivity to noise.
        
        Args:
            activations: Base activations tensor
            noise_level: Standard deviation of noise (relative to activation scale)
            noise_type: Type of noise ("gaussian" or "uniform")
            
        Returns:
            Activations with injected noise
        """
        if noise_type == "gaussian":
            noise = torch.randn_like(activations) * noise_level
        elif noise_type == "uniform":
            noise = (torch.rand_like(activations) * 2 - 1) * noise_level
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        # Scale noise relative to activation magnitude
        activation_scale = activations.std()
        noise = noise * activation_scale
        
        return activations + noise
    
    def get_metadata(
        self,
        input_text: str,
        layer: int,
        token_positions: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Generate metadata for spectrum.
        
        Args:
            input_text: Input text
            layer: Layer number
            token_positions: Token positions used
            
        Returns:
            Metadata dictionary
        """
        return {
            "input_text": input_text,
            "layer": layer,
            "token_positions": token_positions,
            "device": str(self.device)
        }



