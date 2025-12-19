# Quick Start Guide

## Run the Experiment

```bash
# Navigate to specimen
cd /Users/ariahan/Documents/ai-research/experiments/specimens/2024_12_19_hallucination_biopsy_gemma2

# Execute
python protocol.py
```

**Runtime**: 2-5 minutes  
**Output**: Creates `strata/` directory with metrics.parquet and activation tensors

## View Results

### Option 1: Direct File Access

```python
from pathlib import Path
from protocols.storage import SpecimenStorage

storage = SpecimenStorage(Path.cwd())
metrics = storage.read_metrics()
print(metrics)
```

### Option 2: SQL Queries (after indexing)

```bash
# Index the vault first
cd /Users/ariahan/Documents/ai-research/experiments
python scripts/index_vault.py
```

```python
from protocols.query import VaultQuery

vault = VaultQuery()
results = vault.search("""
    SELECT experiment_name, unique_to_hall_count, top_feature_1_words
    FROM exp_2024_12_19_hallucination_biopsy_gemma2
    ORDER BY unique_to_hall_count DESC
""")
print(results)
```

## Expected Output Structure

```
strata/
├── metrics.parquet           # 5 rows × 20 columns (~5KB)
├── fact_activations.zarr/    # (5, 16384) float32 (~160MB)
├── hall_activations.zarr/    # (5, 16384) float32 (~160MB)
└── manifest.msgpack          # Experiment metadata
```

## Key Metrics

- `experiment_name`: Human-readable label
- `unique_to_hall_count`: Number of features unique to hallucination
- `energy_diff`: Total activation difference (hall - fact)
- `top_feature_1..5`: Top biomarker feature IDs
- `top_feature_1..5_words`: Decoded meanings

## Troubleshooting

**Import Error**: `pip install -e .` from experiments root  
**Memory Error**: Close other apps, use CPU device  
**Model Download**: First run downloads ~2GB of models

## Next Steps

1. Analyze results (see README.md for examples)
2. Compare with other specimens
3. Run causal validation experiments

