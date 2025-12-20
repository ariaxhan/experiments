# Engines Directory

This directory contains analysis engines used across experiments. Each engine is self-contained in its own subdirectory with a complete implementation.

## Structure

```
engines/
├── universal_spectroscopy/    # SAE-based feature analysis
│   ├── __init__.py
│   ├── engine.py              # Main UniversalSpectroscopyEngine
│   ├── excitation.py          # ExcitationController
│   ├── sae_adapter.py         # SAE_Adapter
│   ├── interference.py        # InterferenceEngine
│   ├── spectrum.py            # Spectrum data structure
│   └── utils.py               # Utilities
└── [future engines]/          # Add new engines here

```

## Usage

Import engines from their subdirectories:

```python
# Universal Spectroscopy Engine
from engines.universal_spectroscopy import UniversalSpectroscopyEngine, get_device

# Initialize and use
device = get_device()
engine = UniversalSpectroscopyEngine(device=device)
engine.load_model("gemma-2-2b")
engine.load_sae("gemma-2-2b", layer=5)
spectrum = engine.process("Your text here")
purity = engine.calculate_purity(spectrum)
```

## Available Engines

### Universal Spectroscopy Engine

**Purpose**: SAE-based feature analysis treating LLM activations as light spectra

**Components**:
- `UniversalSpectroscopyEngine`: Main orchestrator
- `ExcitationController`: Input processing and activation extraction
- `SAE_Adapter`: SAE loading and feature decomposition
- `InterferenceEngine`: Spectral analysis (purity, drift, absorption)
- `Spectrum`: Data structure for feature activations

**Use Cases**:
- Vector-Native vs Natural Language comparison
- Feature activation analysis
- Semantic drift detection
- Hallucination detection via spectral purity

**Dependencies**: `transformer_lens`, `sae_lens`, `torch`

## Adding New Engines

To add a new engine:

1. Create a new subdirectory: `engines/your_engine_name/`
2. Add `__init__.py` with exports
3. Implement your engine components
4. Update this README with documentation
5. Import in experiments as: `from engines.your_engine_name import YourEngine`

## Design Principles

- **Self-contained**: Each engine has all its dependencies in its subdirectory
- **Modular**: Engines don't depend on each other
- **Documented**: Each engine has clear documentation and usage examples
- **Importable**: Clean import paths from experiments

