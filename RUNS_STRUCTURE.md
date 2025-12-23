# Multiple Runs Per Experiment

## Overview

Experiments can now be run multiple times, with each run preserved in its own timestamped directory. This enables:
- **Reproducibility**: Track changes over time
- **Comparison**: Compare results across runs
- **Iteration**: Test different configurations
- **History**: Complete audit trail of all experiment runs

## New Structure

```
specimens/2024_12_20_vn_comprehensive_experiments/
├── protocol.py              # Reusable protocol (run multiple times)
├── field_notes.md           # Shared notes across all runs
├── manifest.msgpack         # Specimen-level metadata
└── runs/                    # All experiment runs
    ├── 20251222_194653/    # Run 1
    │   ├── metrics.parquet
    │   └── manifest.msgpack
    ├── 20251222_201234/    # Run 2
    │   ├── metrics.parquet
    │   └── manifest.msgpack
    └── 20251223_143022/    # Run 3
        ├── metrics.parquet
        └── manifest.msgpack
```

## Usage

### Running an Experiment

Simply run the protocol multiple times:

```bash
# First run
python specimens/2024_12_20_vn_comprehensive_experiments/protocol.py

# Second run (creates new run automatically)
python specimens/2024_12_20_vn_comprehensive_experiments/protocol.py

# Third run (creates another new run)
python specimens/2024_12_20_vn_comprehensive_experiments/protocol.py
```

Each run creates a new timestamped directory under `runs/`.

### Viewing Results

```bash
# View latest run (default)
python scripts/view_results.py specimens/2024_12_20_vn_comprehensive_experiments

# View all specimens
python scripts/view_results.py --all
```

The viewer automatically shows:
- Current run ID
- Total number of runs
- List of all runs

### Programmatic Access

```python
from protocols.storage import SpecimenStorage
from pathlib import Path

# Initialize storage (creates new run automatically)
storage = SpecimenStorage(Path("specimens/2024_12_20_vn_comprehensive_experiments"))

# List all runs
runs = storage.list_runs()
print(f"Total runs: {len(runs)}")
print(f"Latest: {storage.get_latest_run()}")

# Read from specific run
df = storage.read_metrics(run_id="20251222_194653")

# Read from latest run (default)
df = storage.read_metrics()

# Read from all runs (combined)
df_all = storage.read_all_runs()
```

## Migration

To migrate existing specimens from the old structure:

```bash
# Migrate all specimens
python scripts/migrate_to_runs.py

# Migrate specific specimen
python scripts/migrate_to_runs.py specimens/2024_12_20_vn_comprehensive_experiments

# Dry run (see what would be migrated)
python scripts/migrate_to_runs.py --dry-run
```

## Key Features

### Immutable Runs
- Each run is **immutable** - files within a run cannot be overwritten
- To update results, create a new run
- This ensures complete reproducibility and history

### Automatic Timestamping
- Each run gets a unique timestamp: `YYYYMMDD_HHMMSS`
- Timestamps are based on when the run was created
- Easy to sort chronologically

### Backward Compatibility
- `storage.strata_path` still works (points to current run)
- Existing code continues to work
- Migration script handles legacy data

### Run Metadata
- Each run's manifest includes `run_id` and `run_timestamp`
- Metrics include `run_id` column for easy filtering
- Easy to track which run produced which results

## Benefits

1. **No Data Loss**: Every run is preserved
2. **Easy Comparison**: Compare runs side-by-side
3. **Reproducibility**: Track exactly what changed between runs
4. **Iteration**: Test hypotheses without losing previous results
5. **Audit Trail**: Complete history of all experiment runs

## Example Workflow

```python
# Run experiment multiple times with different configs
for config in configs:
    # Modify config
    CONFIG["some_setting"] = config
    
    # Run experiment (creates new run)
    run_experiment()
    
    # Results automatically saved to new run directory
    # Previous runs remain untouched
```

## Notes

- **Field Notes**: Shared across all runs (in specimen root)
- **Protocol**: Reusable script (can be modified between runs)
- **Manifests**: Each run has its own manifest with run-specific metadata
- **Metrics**: Each run has its own metrics file

