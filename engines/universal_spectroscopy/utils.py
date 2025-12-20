"""●COMPONENT|Ψ:utils|Ω:device_detection_memory_management

Utility functions for device detection, memory management, and caching.
Ported from universal-spectroscopy-engine.
"""

import torch
from contextlib import contextmanager
from typing import Optional, Tuple, Any, Dict
import gc


# Global model cache
_MODEL_CACHE: Dict[str, Any] = {}
_SAE_CACHE: Dict[str, Any] = {}


def get_device() -> torch.device:
    """
    Automatically select best available device.
    
    Priority: MPS (Mac) > CUDA (NVIDIA) > CPU
    
    Returns:
        torch.device: Selected device
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def clear_cache(device: Optional[torch.device] = None) -> None:
    """
    Clear device cache and run garbage collection.
    
    Args:
        device: Device to clear cache for. If None, clears all available caches.
    """
    gc.collect()
    
    if device is None:
        # Clear all available caches
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    else:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()


@contextmanager
def model_context(model: Any, sae: Any):
    """
    Context manager for model and SAE loading/unloading.
    
    Ensures proper cleanup of VRAM after use.
    
    Args:
        model: Model instance (HookedTransformer)
        sae: SAE instance
        
    Yields:
        Tuple of (model, sae)
    """
    try:
        yield model, sae
    finally:
        # Cleanup
        del model, sae
        clear_cache()


def verify_device_compatibility(device: torch.device) -> bool:
    """
    Verify that the requested device is available and compatible.
    
    Args:
        device: Device to verify
        
    Returns:
        True if device is available, False otherwise
    """
    if device.type == "cuda":
        return torch.cuda.is_available()
    elif device.type == "mps":
        return torch.backends.mps.is_available()
    elif device.type == "cpu":
        return True
    else:
        return False


def cache_model(model_name: str, model: Any) -> None:
    """
    Cache a loaded model to avoid reloading.
    
    Args:
        model_name: Name/identifier of the model
        model: Model instance to cache
    """
    _MODEL_CACHE[model_name] = model


def get_cached_model(model_name: str) -> Optional[Any]:
    """
    Retrieve a cached model if available.
    
    Args:
        model_name: Name/identifier of the model
        
    Returns:
        Cached model or None if not found
    """
    return _MODEL_CACHE.get(model_name)


def cache_sae(sae_key: str, sae: Any) -> None:
    """
    Cache a loaded SAE to avoid reloading.
    
    Args:
        sae_key: Unique identifier for the SAE (e.g., "gemma-2-2b-layer5")
        sae: SAE instance to cache
    """
    _SAE_CACHE[sae_key] = sae


def get_cached_sae(sae_key: str) -> Optional[Any]:
    """
    Retrieve a cached SAE if available.
    
    Args:
        sae_key: Unique identifier for the SAE
        
    Returns:
        Cached SAE or None if not found
    """
    return _SAE_CACHE.get(sae_key)


def clear_model_cache() -> None:
    """Clear all cached models and SAEs."""
    global _MODEL_CACHE, _SAE_CACHE
    _MODEL_CACHE.clear()
    _SAE_CACHE.clear()
    clear_cache()

