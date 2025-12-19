"""●COMPONENT|Ψ:demonstration_protocol|Ω:showcase_specimen_vault_pattern

This protocol demonstrates the Specimen Vault pattern using synthetic data.
It shows how to:
1. Generate structured metrics (Parquet format)
2. Generate tensor data (Zarr format)
3. Use SpecimenStorage for persistence
4. Create a complete, reproducible experiment

No external ML dependencies required - uses only numpy for synthetic data generation.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np

from protocols.storage import SpecimenStorage


def generate_synthetic_metrics(num_steps: int = 100) -> Dict[str, List]:
    """●METHOD|input:int|output:dict|operation:create_training_metrics_simulation
    
    Generate synthetic training metrics that simulate a typical ML training run.
    Demonstrates decreasing loss, increasing accuracy, and varying entropy.
    
    Args:
        num_steps: Number of training steps to simulate (default: 100)
    
    Returns:
        Dictionary with columns: step, loss, accuracy, entropy
    """
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Generate step numbers
    steps = list(range(num_steps))
    
    # Generate loss: starts high, decreases with some noise
    # Exponential decay from 2.0 to ~0.1 with Gaussian noise
    loss_base = 2.0 * np.exp(-0.03 * np.array(steps))
    loss_noise = np.random.normal(0, 0.05, num_steps)
    loss = (loss_base + loss_noise).clip(min=0.01).tolist()
    
    # Generate accuracy: starts low, increases (inverse of loss pattern)
    # Sigmoid-like growth from ~0.3 to ~0.95
    accuracy_base = 0.95 - 0.65 / (1 + np.exp((np.array(steps) - 50) / 10))
    accuracy_noise = np.random.normal(0, 0.02, num_steps)
    accuracy = (accuracy_base + accuracy_noise).clip(min=0.0, max=1.0).tolist()
    
    # Generate entropy: starts high, oscillates with decreasing amplitude
    # Represents model confidence increasing over time
    entropy_base = 1.5 * np.exp(-0.01 * np.array(steps)) * (1 + 0.3 * np.sin(np.array(steps) / 5))
    entropy_noise = np.random.normal(0, 0.03, num_steps)
    entropy = (entropy_base + entropy_noise).clip(min=0.01).tolist()
    
    return {
        "step": steps,
        "loss": loss,
        "accuracy": accuracy,
        "entropy": entropy
    }


def generate_synthetic_activations(num_samples: int = 100, hidden_dim: int = 512) -> np.ndarray:
    """●METHOD|input:int_int|output:ndarray|operation:create_synthetic_neural_activations
    
    Generate synthetic neural network activations.
    Simulates intermediate layer outputs from a transformer model.
    
    Args:
        num_samples: Number of activation vectors to generate (default: 100)
        hidden_dim: Dimensionality of each activation vector (default: 512)
    
    Returns:
        Array of shape (num_samples, hidden_dim) with normally distributed values
    """
    # Set seed for reproducibility
    np.random.seed(123)
    
    # Generate activations from standard normal distribution
    # This simulates normalized layer outputs from a neural network
    activations = np.random.randn(num_samples, hidden_dim).astype(np.float32)
    
    return activations


def run_experiment() -> None:
    """●METHOD|input:None|output:None|operation:execute_complete_demonstration_workflow
    
    Main entry point for the demonstration experiment.
    
    This function orchestrates the complete experiment workflow:
    1. Initialize storage for this specimen
    2. Generate synthetic metrics (training curves)
    3. Generate synthetic activations (neural network outputs)
    4. Persist both to optimal formats (Parquet and Zarr)
    5. Print confirmation of successful execution
    """
    print("●PROCESS|operation:demonstration_experiment|phase:starting")
    
    # Initialize storage - automatically creates strata/ directory
    specimen_path = Path(__file__).parent
    storage = SpecimenStorage(specimen_path)
    print(f"  Initialized storage at: {specimen_path}")
    
    # Generate and save metrics
    print("●PROCESS|operation:generating_metrics|phase:execution")
    metrics = generate_synthetic_metrics(num_steps=100)
    storage.write_metrics(metrics)
    print(f"  ✓ Generated and saved metrics ({len(metrics['step'])} steps)")
    print(f"    Format: Parquet (columnar, queryable)")
    print(f"    Location: strata/metrics.parquet")
    
    # Generate and save activations
    print("●PROCESS|operation:generating_activations|phase:execution")
    activations = generate_synthetic_activations(num_samples=100, hidden_dim=512)
    storage.write_tensors("activations", activations)
    print(f"  ✓ Generated and saved activations {activations.shape}")
    print(f"    Format: Zarr (chunked, lazy-loadable)")
    print(f"    Location: strata/activations.zarr")
    
    # Demonstrate lazy loading of tensors
    print("●PROCESS|operation:demonstrating_lazy_loading|phase:execution")
    activations_lazy = storage.read_tensor_lazy("activations")
    print(f"  ✓ Loaded activations handle (lazy, not in memory)")
    print(f"    Type: {type(activations_lazy)}")
    print(f"    Shape: {activations_lazy.shape}")
    print(f"    Chunk shape: {activations_lazy.chunks}")
    
    # Demonstrate metrics reading
    print("●PROCESS|operation:demonstrating_metrics_read|phase:execution")
    metrics_df = storage.read_metrics()
    print(f"  ✓ Loaded metrics as Polars DataFrame")
    print(f"    Shape: {metrics_df.shape}")
    print(f"    Columns: {metrics_df.columns}")
    
    print()
    print("✓ Demonstration complete!")
    print("  Next steps:")
    print("    1. Run: python scripts/index_vault.py")
    print("    2. Query via VaultQuery (see README for examples)")


if __name__ == "__main__":
    run_experiment()
