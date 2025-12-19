# Hallucination Biopsy: SAE-Based Differential Diagnosis

**Specimen ID**: `2024_12_19_hallucination_biopsy_gemma2`  
**Domain**: Interpretability  
**Method**: SAE Analysis  
**Status**: Active

## Overview

This specimen performs differential diagnosis on language model hallucinations using Sparse Autoencoder (SAE) analysis. It identifies specific neural features ("biomarkers") that activate uniquely during hallucinated outputs, providing mechanistic insight into how models generate false information.

## Quick Start

### Prerequisites

```bash
# Install dependencies (if not already installed)
pip install transformer_lens sae_lens torch numpy polars zarr msgpack
```

### Run the Experiment

```bash
# Navigate to specimen directory
cd /Users/ariahan/Documents/ai-research/experiments/specimens/2024_12_19_hallucination_biopsy_gemma2

# Execute protocol
python protocol.py
```

**Expected Runtime**: 2-5 minutes (depending on device)  
**Memory Required**: ~4GB  
**Device Support**: MPS (Apple Silicon), CUDA, or CPU

### Output

The experiment generates:

1. **strata/metrics.parquet**: Comparative statistics for 5 fact/hallucination pairs
   - Experiment metadata (names, texts)
   - Feature activation counts
   - Energy differences
   - Top 5 biomarker features per experiment
   - Decoded feature meanings (words)

2. **strata/fact_activations.zarr**: Full SAE activations for factual texts
   - Shape: (5, 16384)
   - Format: Zarr (chunked, lazy-loadable)

3. **strata/hall_activations.zarr**: Full SAE activations for hallucinated texts
   - Shape: (5, 16384)
   - Format: Zarr (chunked, lazy-loadable)

4. **strata/manifest.msgpack**: Experiment metadata and configuration

## Experiment Design

### Fact/Hallucination Pairs

1. **Geography Teleportation**: Eiffel Tower (Paris → Rome)
2. **Geography Teleportation 2**: Golden Gate Bridge (San Francisco → New York)
3. **Historical Anachronism**: Shakespeare (Hamlet → Star Wars)
4. **Biological Impossibility**: Dogs (four legs → two wings)
5. **Mathematical Inversion**: Five vs Two (greater → smaller)

### Methodology

For each pair:
1. Extract SAE feature activations from both texts (layer 5, last token)
2. Identify features unique to hallucination (active in hall, zero in fact)
3. Rank by activation magnitude (loudest features first)
4. Decode feature meanings via vocabulary projection

## Querying Results

### Index the Vault

After running the experiment, index it into the vault:

```bash
cd /Users/ariahan/Documents/ai-research/experiments
python scripts/index_vault.py
```

### SQL Queries

```python
from protocols.query import VaultQuery

vault = VaultQuery()

# View all experiments
results = vault.search("""
    SELECT experiment_name, unique_to_hall_count, energy_diff
    FROM exp_2024_12_19_hallucination_biopsy_gemma2
    ORDER BY unique_to_hall_count DESC
""")
print(results)

# Find experiments with high energy difference
results = vault.search("""
    SELECT experiment_name, energy_diff, top_feature_1, top_feature_1_words
    FROM exp_2024_12_19_hallucination_biopsy_gemma2
    WHERE energy_diff > 0
""")
print(results)
```

### Programmatic Access

```python
from pathlib import Path
from protocols.storage import SpecimenStorage

# Load specimen
specimen_path = Path("specimens/2024_12_19_hallucination_biopsy_gemma2")
storage = SpecimenStorage(specimen_path)

# Read metrics
metrics_df = storage.read_metrics()
print(metrics_df)

# Read activations (lazy)
fact_acts = storage.read_tensor_lazy("fact_activations")
hall_acts = storage.read_tensor_lazy("hall_activations")

# Compute differential activation
import numpy as np
diff = hall_acts[:] - fact_acts[:]

# Find most differentially activated features
top_features = np.argsort(np.abs(diff).sum(axis=0))[-10:]
print(f"Top 10 differential features: {top_features}")
```

## Analysis Examples

### Identify Common Biomarkers

```python
import polars as pl

# Load metrics
metrics = storage.read_metrics()

# Collect all top features
all_features = []
for i in range(1, 6):
    col = f"top_feature_{i}"
    all_features.extend(metrics[col].to_list())

# Find features that appear multiple times
from collections import Counter
feature_counts = Counter(f for f in all_features if f != -1)
common_biomarkers = feature_counts.most_common(5)
print(f"Features appearing in multiple experiments: {common_biomarkers}")
```

### Visualize Activation Differences

```python
import matplotlib.pyplot as plt
import numpy as np

# Load activations
fact_acts = storage.read_tensor_lazy("fact_activations")[:]
hall_acts = storage.read_tensor_lazy("hall_activations")[:]

# Compute difference
diff = hall_acts - fact_acts

# Plot heatmap
plt.figure(figsize=(12, 6))
plt.imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
plt.colorbar(label='Activation Difference (Hall - Fact)')
plt.xlabel('Feature Index')
plt.ylabel('Experiment')
plt.title('Differential Feature Activations')
plt.yticks(range(5), metrics['experiment_name'].to_list())
plt.tight_layout()
plt.savefig('activation_heatmap.png', dpi=150)
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'protocols'`:

```bash
# Install the experiments package in development mode
cd /Users/ariahan/Documents/ai-research/experiments
pip install -e .
```

### Memory Issues

If you run out of memory:
- Close other applications
- Use CPU instead of MPS/CUDA (slower but more stable)
- Reduce batch size (modify `experiments` list in protocol.py)

### Model Download

First run will download models (~2GB):
- Gemma-2-2b model weights
- GemmaScope SAE weights

Subsequent runs use cached models.

## Next Steps

1. **Causal Validation**: Run ablation studies to verify biomarker features causally affect hallucination
2. **Cross-Layer Analysis**: Repeat at multiple layers to track feature evolution
3. **Larger Scale**: Expand to 50-100 fact/hallucination pairs
4. **Cross-Model**: Compare with Gemma-2-9b or Llama models

## References

- **GemmaScope**: [Neuronpedia](https://www.neuronpedia.org/gemma-scope)
- **Specimen Vault Pattern**: See main experiments/README.md
- **SAE Analysis**: Sparse Autoencoder interpretability methods

## Metadata

**Created**: 2024-12-19  
**Tags**: gemma-2-2b, layer_5, hallucination_detection, sae_diagnosis  
**Dependencies**: transformer_lens, sae_lens, torch, numpy, polars, zarr, msgpack  
**Storage**: ~320MB (metrics + tensors)

