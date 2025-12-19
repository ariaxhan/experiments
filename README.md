# Specimen Vault: ML Experiment Archive

A local-first, queryable system for ML research experiments using the **Specimen Vault** pattern.

## ●PHILOSOPHY|pattern:specimen_collection

Like natural history specimens preserved in museum vaults, each experiment in this system is:

- **Preserved**: Immutable records with optimal storage formats
- **Cataloged**: Indexed with rich metadata (domain, method, model, tags)
- **Queryable**: DuckDB SQL across all specimens instantly
- **Reproducible**: Complete methodology and dependencies recorded

**The Analogy**:
- Each experiment = Individual specimen (timestamped, never modified)
- Results = Archived in optimal formats (Parquet for metrics, Zarr for tensors)
- Catalog = Master database enabling cross-experiment analysis
- Protocols = Shared methodology for specimen collection

This pattern prioritizes **local-first operation** (no cloud dependencies), **format optimization** (Parquet/Zarr/msgpack), and **scientific reproducibility**.

---

## ●ARCHITECTURE|purpose:format_selection

**Why Parquet for metrics?**
Columnar storage format optimized for analytical queries. Compresses well, maintains schemas, and DuckDB can query directly without loading into memory. Perfect for training curves (loss, accuracy) and tabular experiment logs.

**Why Zarr for tensors?**
Chunked array storage enabling lazy loading. Load only the slices you need from large activation matrices or embeddings. Supports compression and is numpy-compatible. Avoids memory explosions when working with GB-scale tensor data.

**Why msgpack for config?**
Binary serialization that's faster and more compact than JSON, with proper type support. Human-readable when decoded, but efficient on disk. Ideal for manifests and configuration files.

**Why DuckDB for catalog?**
In-process SQL database (no server needed). Queries Parquet files directly via zero-copy reads. Perfect for local-first cross-experiment analysis. Supports complex queries (JOINs, aggregations) on experiment metadata and metrics.

---

## ●QUICKSTART|workflow:experiment_lifecycle

### 1. Installation

```bash
# Clone repository
git clone https://github.com/ariaxhan/experiments.git
cd experiments

# Install in development mode
pip install -e .
```

**Dependencies**: Python >=3.8, polars, zarr, msgpack, duckdb, numpy

### 2. Run the Example

```bash
# Execute demonstration experiment (synthetic data, no ML dependencies)
python specimens/example_2024_12_19_synthetic_demo/protocol.py

# Output creates:
# - strata/metrics.parquet (training curves: 100 steps)
# - strata/activations.zarr (synthetic activations: 100x512)
```

**Note**: Generated artifacts (`*.zarr/`, `vault.duckdb`) are excluded from git (see `.gitignore`). This is by design - you generate them by running experiments and indexing.

### 3. Create Your Own Specimen

```bash
# Scaffold new experiment directory
python scripts/create_specimen.py \
  --id 2024_12_19_my_experiment \
  --domain interpretability \
  --method sae_analysis \
  --tags gpt2,layer_12

# This creates:
# specimens/2024_12_19_my_experiment/
# ├── manifest.msgpack          # Metadata
# ├── protocol.py               # Experiment script template
# ├── field_notes.md            # Documentation template
# └── strata/                   # Results directory (empty initially)
```

### 4. Implement Your Experiment

Edit `protocol.py` in your specimen directory:

```python
from pathlib import Path
from protocols.storage import SpecimenStorage

def run_experiment():
    storage = SpecimenStorage(Path(__file__).parent)
    
    # Generate metrics (dict with columns as keys, lists as values)
    metrics = {
        "step": [0, 1, 2, 3],
        "loss": [0.5, 0.3, 0.2, 0.1]
    }
    storage.write_metrics(metrics)
    
    # Generate tensors (numpy arrays)
    import numpy as np
    activations = np.random.randn(100, 768)
    storage.write_tensors("activations", activations)

if __name__ == "__main__":
    run_experiment()
```

Then run it:

```bash
cd specimens/2024_12_19_my_experiment
python protocol.py
```

### 5. Index the Vault

```bash
# Rebuild catalog from all specimen manifests
python scripts/index_vault.py

# Output: vault.duckdb with catalog table and per-specimen SQL views
```

### 6. Query Your Experiments

```python
from protocols.query import VaultQuery

# Initialize query engine (connects to vault.duckdb)
vault = VaultQuery()

# Search catalog
specimens = vault.search("SELECT * FROM catalog WHERE domain = 'interpretability'")
print(specimens)

# Query specific specimen metrics
results = vault.search("SELECT * FROM exp_example_2024_12_19_synthetic_demo WHERE loss < 0.5")
print(results)
```

---

## ●TAXONOMY|structure:classification

Valid domain and method combinations for experiment classification:

| **Domain** | **Description** | **Valid Methods** |
|------------|-----------------|-------------------|
| `interpretability` | Understanding model internals | `sae_analysis`, `probe_training`, `ablation_study` |
| `training` | Model training and optimization | `fine_tuning`, `ablation_study` |
| `benchmarking` | Evaluation and comparison | `ablation_study` |
| `alignment` | Safety and capability research | `fine_tuning`, `probe_training`, `ablation_study` |

| **Method** | **Description** |
|------------|-----------------|
| `sae_analysis` | Sparse autoencoder analysis of model representations |
| `probe_training` | Training linear probes to detect learned features |
| `ablation_study` | Removing components to measure impact |
| `fine_tuning` | Continued training on specific data or tasks |

**Usage**: These taxonomies enforce structure for searchability and organization. Choose the most specific domain and method for your experiment.

---

## ●STRUCTURE|layout:directory_tree

```
experiments/
├─ specimens/                          # Experiment collection (immutable, timestamped)
│  ├─ example_2024_12_19_synthetic_demo/
│  │  ├─ manifest.msgpack              # Metadata: domain, method, tags, dependencies
│  │  ├─ protocol.py                   # Executable experiment script
│  │  ├─ field_notes.md                # Observations and methodology
│  │  └─ strata/                       # Results directory
│  │     ├─ metrics.parquet            # Training curves, structured logs
│  │     └─ activations.zarr/          # Tensor data (chunked arrays)
│  └─ [more specimens...]
│
├─ protocols/                          # Shared utilities (importable Python package)
│  ├─ __init__.py
│  ├─ storage.py                       # SpecimenStorage: write/read Parquet/Zarr
│  └─ query.py                         # VaultQuery: SQL search across specimens
│
├─ scripts/                            # Automation tools (executable)
│  ├─ create_specimen.py               # Scaffold new experiment directory
│  └─ index_vault.py                   # Rebuild vault.duckdb catalog
│
├─ vault.duckdb                        # Master catalog (auto-generated, don't edit)
├─ pyproject.toml                      # Python package configuration
└─ README.md                           # This file
```

**Key Principles**:
- **specimens/**: Append-only. Never modify existing specimens. Create new ones for variations.
- **strata/**: Name comes from geology (layers of data). Auto-created by SpecimenStorage.
- **vault.duckdb**: Rebuilt by `index_vault.py`. Queries specimens without copying data.

---

## ●EXAMPLES|queries:sql_patterns

### Example 1: Find All Specimens by Domain

```python
from protocols.query import VaultQuery

vault = VaultQuery()
results = vault.search("""
    SELECT specimen_id, created, method, tags 
    FROM catalog 
    WHERE domain = 'interpretability'
    ORDER BY created DESC
""")
print(results)
```

### Example 2: Search by Tag

```python
vault = VaultQuery()
results = vault.search("""
    SELECT specimen_id, domain, method 
    FROM catalog 
    WHERE array_contains(tags, 'gpt2')
""")
print(results)
```

### Example 3: Find Recent Experiments

```python
vault = VaultQuery()
results = vault.search("""
    SELECT specimen_id, created, domain, method 
    FROM catalog 
    WHERE created >= '2024-12-01'
    ORDER BY created DESC
    LIMIT 10
""")
print(results)
```

### Example 4: Query Specific Specimen Metrics

```python
vault = VaultQuery()

# Each specimen gets a view: exp_{specimen_id}
results = vault.search("""
    SELECT step, loss, accuracy 
    FROM exp_example_2024_12_19_synthetic_demo 
    WHERE step >= 50
""")
print(results)
```

### Example 5: Aggregate Across Multiple Specimens

```python
vault = VaultQuery()

# Find all interpretability experiments with final loss < 0.2
results = vault.search("""
    WITH specimen_ids AS (
        SELECT specimen_id 
        FROM catalog 
        WHERE domain = 'interpretability'
    )
    SELECT 
        'exp_' || specimen_id AS view_name,
        MAX(step) AS final_step
    FROM specimen_ids
    -- Note: In practice you'd UNION views or use DuckDB list queries
    LIMIT 10
""")
print(results)
```

### Example 6: Compare Metrics Across Experiments

```python
# Advanced: Query multiple specimen views simultaneously
vault = VaultQuery()

results = vault.search("""
    SELECT 
        'example' AS experiment,
        AVG(loss) AS mean_loss,
        MIN(loss) AS min_loss
    FROM exp_example_2024_12_19_synthetic_demo
""")
print(results)
```

---

## ●SECTION|troubleshooting:common_issues

### Issue: "No module named 'protocols'"

**Cause**: Package not installed in development mode.

**Solution**:
```bash
pip install -e .
```

### Issue: "Failed to create strata directory"

**Cause**: Insufficient write permissions or invalid path.

**Solution**:
- Check that you're running from the correct directory
- Verify write permissions: `ls -ld specimens/`
- Ensure parent directories exist

### Issue: "Specimen ID already exists"

**Cause**: Trying to create a specimen with a duplicate ID.

**Solution**:
- Choose a different date/descriptor: `2024_12_19_experiment_v2`
- Never modify existing specimens - create new ones instead

### Issue: "Missing manifest" warning during indexing

**Cause**: Incomplete specimen directory (no `manifest.msgpack`).

**Solution**:
- Use `create_specimen.py` to scaffold specimens properly
- Manually add manifest if created outside of scaffolding tool
- Check for `.gitkeep` files that shouldn't be specimens

### Issue: "Corrupt manifest" error

**Cause**: Invalid msgpack format or encoding issues.

**Solution**:
```python
import msgpack
from pathlib import Path

# Read and validate
with open("specimens/your_specimen/manifest.msgpack", "rb") as f:
    data = msgpack.unpack(f)
    print(data)  # Inspect structure
```

### Issue: "No valid specimens found to index"

**Cause**: Empty `specimens/` directory or all specimens lack manifests.

**Solution**:
- Run the example: `python specimens/example_2024_12_19_synthetic_demo/protocol.py`
- Create a new specimen: `python scripts/create_specimen.py --id test --domain interpretability --method sae_analysis`

### Issue: Path errors when running protocol.py

**Cause**: Running from wrong directory or incorrect Path usage.

**Solution**:
- Always run from specimen directory: `cd specimens/my_experiment && python protocol.py`
- Or run from root with full path: `python specimens/my_experiment/protocol.py`
- Use `Path(__file__).parent` in protocols for specimen-relative paths

---

## ●SECTION|development:contributing

### Code Style

- **Type hints**: Required on all function signatures
- **Paths**: Use `pathlib.Path`, never string paths
- **Error handling**: Wrap I/O operations with try/except, raise RuntimeError with context
- **Formats**: Polars over pandas (unless legacy constraints)

### Vector Native Documentation

This repository uses Vector Native (VN) format for structured documentation:

- `●SECTION|id`: Major sections
- `●COMPONENT|Ψ:role|Ω:goal`: Component docstrings
- `●METHOD|input:type|output:type|operation:description`: Method docstrings
- `●LOG|Δ:action|metadata`: Event logs in field_notes.md

Plain English for user-facing docs (like this README), VN for code and internal docs.

### Running Tests

Currently no test suite. Validation via:
- Running example specimen: `python specimens/example_2024_12_19_synthetic_demo/protocol.py`
- Testing indexing: `python scripts/index_vault.py`
- Manual queries via VaultQuery

---

## ●SECTION|license

MIT License - see LICENSE file for details
