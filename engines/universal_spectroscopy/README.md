# Universal Spectroscopy Engine (USE)

A framework treating LLM activations as light spectra to measure semantic drift and hallucinations.

## Overview

The Universal Spectroscopy Engine (USE) is a spectroscopy-inspired framework that treats Large Language Model (LLM) activations as physical light spectra. This novel approach enables the diagnosis of semantic drift, hallucinations, and model blindness locally, providing interpretable insights into model behavior.

Following the "Physics of Meaning" metaphor:
- **Light Source** → User Input / Prompt
- **Material** → The LLM (e.g., Gemma-2-2B)
- **Prism** → Sparse Autoencoder (SAE)
- **Spectrum** → Feature Activations (Indices & Magnitudes)
- **Spectral Lines** → Monosemantic Features (Concepts)
- **Thermal Noise** → Polysemantic/Dense Activations

## Core Hypotheses

- **H1: Spectral Purity** - Hallucinations manifest as low spectral purity (high entropy/noise, few distinct peaks)
- **H2: Doppler Shift** - Semantic meaning "redshifts" (generalizes) or "blueshifts" (distorts) through agent chains
- **H3: Absorption** - Missing features indicate ignored instructions (model blindness)

## Installation

This engine is part of the experiments repository. Install the repository:

```bash
cd experiments
pip install -e .
```

**Dependencies**:
- `transformer_lens`: For working with transformer models
- `sae_lens`: For loading sparse autoencoders
- `torch`: PyTorch for deep learning
- `numpy`: Numerical operations

## Getting Started

### Import the Engine

```python
from engines.universal_spectroscopy import UniversalSpectroscopyEngine, get_device
```

### Basic Usage

```python
# Initialize engine
device = get_device()
engine = UniversalSpectroscopyEngine(device=device)

# Load model (downloads automatically on first use)
print("Loading model...")
engine.load_model("gemma-2-2b")

# Load SAE from Gemma-Scope (downloads automatically)
print("Loading SAE...")
engine.load_sae(
    model_name="gemma-2-2b",
    layer=5,
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_5/width_16k/canonical"
)

# Or use auto-detection (simpler)
engine.load_sae("gemma-2-2b", layer=5)

# Process input text
print("Processing input...")
input_text = "The cat sat on the mat."
spectrum = engine.process(input_text)

print(f"Spectrum: {spectrum}")
print(f"Number of active features: {len(spectrum)}")
print(f"Top 10 features: {spectrum.get_top_features(k=10)}")
```

### Test Hypotheses

```python
# H1: Calculate Spectral Purity (Hallucination Detection)
purity = engine.calculate_purity(spectrum)
print(f"Spectral Purity: {purity:.4f}")
if purity < 0.3:
    print("⚠️  Warning: Low spectral purity - possible hallucination")

# H2: Calculate Semantic Drift (Compare two spectra)
input_spec = engine.process("The cat sat on the mat.")
output_spec = engine.process("The feline rested on the rug.")
drift = engine.calculate_drift(input_spec, output_spec)
print(f"Semantic Drift: {drift:.4f}")
if drift > 0.5:
    print("⚠️  Significant semantic drift detected")

# H3: Detect Absorption (Model Blindness)
absorbed = engine.detect_absorption(input_spec, output_spec)
if absorbed:
    print(f"⚠️  Model ignored {len(absorbed)} features: {absorbed[:10]}...")

# Cleanup
engine.cleanup()
```

### Advanced Usage with Context Manager

```python
from engines.universal_spectroscopy import UniversalSpectroscopyEngine

# Use context manager for automatic cleanup
with UniversalSpectroscopyEngine() as engine:
    engine.load_model("gemma-2-2b")
    engine.load_sae("gemma-2-2b", layer=5)
    
    spectrum = engine.process("Your text here")
    purity = engine.calculate_purity(spectrum)
    
    # Cleanup happens automatically
```

## Project Structure

```
engines/universal_spectroscopy/
├── __init__.py          # Package exports
├── engine.py            # UniversalSpectroscopyEngine (main class)
├── excitation.py        # ExcitationController (The Slit)
├── sae_adapter.py       # SAE_Adapter (The Prism)
├── interference.py      # InterferenceEngine (The Detector)
├── spectrum.py          # Spectrum data class
├── utils.py             # Device detection, caching, helpers
└── README.md            # This file
```

## Components

### UniversalSpectroscopyEngine

Main orchestration class that coordinates all components.

**Key Methods**:
- `load_model(model_name)`: Load transformer model
- `load_sae(model_name, layer, ...)`: Load SAE for specified layer
- `process(input_text)`: Process text and return Spectrum
- `calculate_purity(spectrum)`: H1 - Calculate spectral purity
- `calculate_drift(input_spec, output_spec)`: H2 - Measure semantic drift
- `detect_absorption(input_spec, output_spec)`: H3 - Detect missing features
- `cleanup()`: Free resources

### ExcitationController (The Slit)

Manages input formatting and feature steering:
- `process()`: Extract activations from model layers
- `monochromatic_steering()`: Force specific features
- `pulse_train()`: Inject noise for robustness testing
- `get_metadata()`: Generate metadata for spectra

### SAE_Adapter (The Prism)

Loads SAEs and normalizes outputs:
- `load_sae()`: Load SAE for model and layer (auto-detects release)
- `decompose()`: Convert activations to Spectrum objects
- `normalize()`: Standardize spectrum format
- `is_loaded()`: Check if SAE is loaded

### InterferenceEngine (The Detector)

Mathematical analysis module:
- `calculate_purity()`: H1 - Spectral purity for hallucination detection
- `calculate_drift()`: H2 - Semantic drift measurement
- `detect_absorption()`: H3 - Missing features detection

### Spectrum

Data class representing feature activations:
- `wavelengths`: Feature indices (which features are active)
- `intensities`: Activation magnitudes (how strongly)
- `model_name`: Model that generated the activations
- `layer`: Layer number
- `metadata`: Additional metadata (token positions, etc.)

**Key Methods**:
- `get_top_features(k)`: Get top k features by intensity
- `get_top_intensities(k)`: Get top k intensities
- `filter_by_threshold(threshold)`: Filter features above threshold
- `__len__()`: Number of active features

### Utils

Utility functions:
- `get_device()`: Auto-detect best device (MPS/CUDA/CPU)
- `clear_cache()`: Clear device cache and run garbage collection
- `model_context()`: Context manager for model/SAE loading
- `cache_model()` / `get_cached_model()`: Model caching
- `cache_sae()` / `get_cached_sae()`: SAE caching

## Usage in Experiments

This engine is designed to be used within the experiments repository's Specimen Vault pattern:

```python
from pathlib import Path
from protocols.storage import SpecimenStorage
from engines.universal_spectroscopy import UniversalSpectroscopyEngine, get_device

def run_experiment():
    # Initialize storage
    storage = SpecimenStorage(Path(__file__).parent)
    
    # Initialize engine
    device = get_device()
    engine = UniversalSpectroscopyEngine(device=device)
    
    # Load model and SAE
    engine.load_model("gemma-2-2b")
    engine.load_sae("gemma-2-2b", layer=5)
    
    # Process text
    spectrum = engine.process("Your text here")
    purity = engine.calculate_purity(spectrum)
    
    # Save results to Specimen Vault
    storage.write_metrics({
        "purity": [purity],
        "num_features": [len(spectrum)]
    })
    
    # Cleanup
    engine.cleanup()
```

## Supported Models and SAEs

### Models
- **Gemma-2-2B**: Fully supported
- **Llama-3-8B**: Supported (requires appropriate SAE)
- Other models via `transformer_lens`: Check compatibility

### SAEs
- **Gemma-Scope**: Auto-detected for Gemma-2-2B
  - Release: `gemma-scope-2b-pt-res-canonical`
  - Layers: 0-25
  - Width: 16k (default)
- **Other SAEs**: Specify `release` and `sae_id` explicitly

## Troubleshooting

### Device Issues

**Problem**: CUDA not available on Mac  
**Solution**: Engine automatically detects MPS (Apple Silicon). If you see CUDA errors, ensure you're using the latest PyTorch with MPS support.

**Problem**: Out of memory errors  
**Solution**: 
- Use smaller models (e.g., gemma-2-2b instead of larger variants)
- Use `engine.cleanup()` or context managers
- Implement model offloading if needed

### SAE Loading Issues

**Problem**: SAE not found or download fails  
**Solution**: 
- Ensure you have internet connection (SAEs download from Hugging Face on first use)
- Check model name and layer number are correct
- For Gemma-2-2B, use layers 0-25
- Verify `sae_lens` is installed: `pip install sae-lens`

### Import Errors

**Problem**: `ModuleNotFoundError` for engines  
**Solution**: 
```bash
# Reinstall the package
cd experiments
pip install -e .
```

**Problem**: `ModuleNotFoundError` for transformer_lens or sae_lens  
**Solution**: 
```bash
pip install transformer_lens sae-lens
```

## Examples

See the experiments repository for complete examples:
- `specimens/2024_12_20_vn_comprehensive_experiments/`: Vector-Native vs Natural Language comparison
- `specimens/2024_12_19_hallucination_biopsy_gemma2/`: Hallucination detection via spectral purity

## Performance Considerations

- **Memory**: Loading both model and SAE is VRAM-heavy. Use cleanup methods.
- **Caching**: Models and SAEs are cached automatically to avoid reloading.
- **Device**: MPS (Mac) is automatically preferred over CUDA when available.

## Next Steps

1. **Run Example Experiments**: See `specimens/` directory for complete examples
2. **Test Hypotheses**: Test H1 (Spectral Purity), H2 (Semantic Drift), H3 (Absorption)
3. **Explore Different Layers**: Try different layers (0-25 for Gemma-2-2B)
4. **Create Visualizations**: Implement spectral barcode rendering
5. **Add New Experiments**: Use this engine in your own specimens

## License

MIT (same as original Universal Spectroscopy Engine)

