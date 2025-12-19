●LOG|specimen:example_2024_12_19_synthetic_demo|created:2024-12-19T23:12:54Z

# Field Notes: example_2024_12_19_synthetic_demo

●SECTION|purpose
This specimen demonstrates the Specimen Vault pattern for ML research tracking. It serves as a pedagogical example showing how to structure experiments, persist data in optimal formats, and maintain reproducibility.

●SECTION|methodology
- **Domain**: interpretability
- **Method**: sae_analysis
- **Model**: synthetic (no real ML model - demonstration only)
- **Tags**: demo, example, synthetic

This is a synthetic demonstration that requires no external ML dependencies (torch, transformers, etc.). It uses only numpy to generate realistic-looking training metrics and activation patterns.

●SECTION|what_this_demonstrates

**1. Metrics Storage (Parquet)**
- Columnar format optimized for analytical queries
- 100 simulated training steps with loss, accuracy, and entropy curves
- Loss decreases exponentially with noise (realistic training pattern)
- Accuracy increases following sigmoid growth
- Entropy shows oscillating decay (confidence increasing)

**2. Tensor Storage (Zarr)**
- Chunked array format enabling lazy loading
- Shape (100, 512): 100 activation vectors, 512 dimensions each
- Normally distributed values simulating neural network layer outputs
- Demonstrates how to avoid loading full arrays into memory

**3. Storage API Usage**
- `SpecimenStorage` initialization and directory creation
- `write_metrics()` for structured tabular data
- `write_tensors()` for multi-dimensional arrays
- `read_metrics()` returning Polars DataFrame
- `read_tensor_lazy()` returning Zarr array handle (no memory load)

●SECTION|observations

**Reproducibility**
All random generation uses fixed seeds (seed=42 for metrics, seed=123 for activations) ensuring exact reproducibility across runs. This is critical for scientific experiments.

**Format Choices**
- **Not CSV**: No schema, inefficient for large data
- **Not pickle**: Not portable, security risks
- **Yes Parquet**: Type-safe, columnar, queryable by DuckDB
- **Yes Zarr**: Lazy loading, chunked access, numpy-compatible

**Performance**
- Metrics file: ~5KB (highly compressed via Parquet)
- Activations: ~200KB (chunked 100x512 float32 array)
- No loading delays - demonstrating instant execution on synthetic data

●SECTION|results

**Generated Artifacts**
```
strata/
├── manifest.msgpack    # Metadata: domain, method, model, tags, dependencies
├── metrics.parquet     # Training curves: step, loss, accuracy, entropy
└── activations.zarr/   # Synthetic neural activations (100, 512)
```

**Expected Behavior**
When running `python protocol.py`:
1. Creates strata/ directory automatically
2. Writes metrics.parquet (100 rows, 4 columns)
3. Writes activations.zarr (chunked array)
4. Demonstrates lazy loading (returns handle, not data)
5. Shows metrics as Polars DataFrame

**Indexing**
After running, execute `python scripts/index_vault.py` to:
- Add specimen to vault.duckdb catalog
- Create SQL view: `exp_example_2024_12_19_synthetic_demo`
- Enable cross-experiment queries

●SECTION|next_steps

**For Real Experiments**
1. Replace synthetic data generation with actual model inference
2. Add more tensor types (embeddings, gradients, attention weights)
3. Include configuration files (hyperparameters, model architecture)
4. Record GPU memory usage, wall-clock time
5. Store multiple checkpoints over training

**For Exploration**
1. Modify `generate_synthetic_metrics()` to simulate different training dynamics
2. Add more metrics columns (learning_rate, gradient_norm, perplexity)
3. Increase activation dimensions to match real models (e.g., 768, 1024, 2048)
4. Experiment with different Zarr chunk sizes for performance tuning

**For Learning**
1. Inspect the generated Parquet file with `parquet-tools` or DuckDB
2. Explore Zarr array metadata: `zarr.open_array('strata/activations.zarr', 'r').info`
3. Query metrics via SQL after indexing: `SELECT * FROM exp_example_2024_12_19_synthetic_demo WHERE loss < 0.5`
4. Compare file sizes: Parquet vs CSV, Zarr vs numpy .npy
