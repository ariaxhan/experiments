# experiments
ML experiment vault - in progress

●SECTION|architecture
Specimen Vault pattern for ML research with local-first, queryable experiment tracking.

●SECTION|quickstart

Create a new experiment:
```bash
./scripts/create_specimen.py \
  --id 2024_12_19_my_experiment \
  --domain interpretability \
  --method sae_analysis \
  --tags gpt2,layer_12
```

Index all experiments:
```bash
./scripts/index_vault.py
```

●SECTION|structure
```
experiments/
├─ specimens/          # Individual experiments (timestamped, immutable)
├─ protocols/          # Shared utilities (storage, query, analysis)
├─ scripts/            # Automation (scaffolding, indexing)
└─ vault.duckdb        # Master catalog (auto-generated)
```
