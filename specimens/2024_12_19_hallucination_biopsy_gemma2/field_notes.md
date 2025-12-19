●LOG|specimen:2024_12_19_hallucination_biopsy_gemma2|created:2024-12-19

# Field Notes: Hallucination Biopsy via SAE Analysis

●SECTION|purpose

This experiment performs differential diagnosis on language model hallucinations using Sparse Autoencoder (SAE) analysis. The goal is to identify specific neural features that activate uniquely during hallucinated outputs, providing mechanistic insight into how models generate false information.

**Research Question**: Can we detect "biomarker" features that distinguish hallucinated text from factual text at the activation level?

●SECTION|methodology

## Domain & Classification
- **Domain**: interpretability
- **Method**: sae_analysis
- **Tags**: gemma-2-2b, layer_5, hallucination_detection, sae_diagnosis

## Experimental Design

### Instruments
1. **Subject Model**: Gemma-2-2b (Google's 2B parameter language model)
   - Architecture: Transformer decoder
   - Configuration: 26 layers, d_model=2304, vocab_size=256128
   
2. **Microscope**: GemmaScope SAE (gemma-scope-2b-pt-res-canonical)
   - Target Layer: Layer 5 residual stream (blocks.5.hook_resid_post)
   - Feature Width: 16,384 features (16k)
   - Purpose: Decompose activations into interpretable features

### Protocol

**Phase 1: Biopsy Extraction**
For each text input:
1. Tokenize and run through Gemma-2-2b
2. Extract activation vector from last token position at layer 5
3. Apply SAE encoder to get sparse feature activations
4. Record active feature indices and magnitudes

**Phase 2: Differential Diagnosis**
For each fact/hallucination pair:
1. Extract biopsies from both texts
2. Identify features unique to hallucination (active in hall, zero in fact)
3. Rank by activation magnitude (loudest features first)
4. Decode feature meanings via vocabulary projection

**Phase 3: Feature Translation**
For each biomarker feature:
1. Extract feature direction from SAE decoder (W_dec)
2. Project through model's unembedding layer
3. Identify top-k words with highest logits
4. Interpret semantic meaning

### Experiment Pairs

Five carefully designed fact/hallucination pairs covering different error types:

1. **Geography Teleportation**: Eiffel Tower (Paris → Rome)
2. **Geography Teleportation 2**: Golden Gate Bridge (San Francisco → New York)
3. **Historical Anachronism**: Shakespeare (Hamlet → Star Wars)
4. **Biological Impossibility**: Dogs (four legs → two wings)
5. **Mathematical Inversion**: Five vs Two (greater → smaller)

Each pair maintains:
- Identical sentence structure
- Similar token count
- Only the critical fact differs

●SECTION|data_structures

## Metrics (Parquet)

Columnar data for cross-experiment analysis:

| Column | Type | Description |
|--------|------|-------------|
| experiment_index | int | Sequential experiment number (1-5) |
| experiment_name | str | Human-readable experiment label |
| fact_text | str | Ground truth input text |
| hallucination_text | str | Hallucinated variant |
| fact_total_active | int | Number of active features in fact |
| hall_total_active | int | Number of active features in hallucination |
| fact_energy | float | Sum of feature magnitudes (fact) |
| hall_energy | float | Sum of feature magnitudes (hallucination) |
| energy_diff | float | hall_energy - fact_energy |
| unique_to_hall_count | int | Features active only in hallucination |
| missing_from_hall_count | int | Features active only in fact |
| top_feature_1..5 | int | Top 5 biomarker feature IDs (by magnitude) |
| top_feature_1..5_words | str | Comma-separated top words for each feature |

## Tensors (Zarr)

High-dimensional activation data:

- **fact_activations.zarr**: Shape (5, 16384)
  - Full SAE feature activations for all fact texts
  - Sparse representation (most values near zero)
  - Chunked for efficient lazy loading

- **hall_activations.zarr**: Shape (5, 16384)
  - Full SAE feature activations for all hallucination texts
  - Enables post-hoc analysis of any feature
  - Supports differential activation heatmaps

●SECTION|observations

## Initial Findings

### Activation Patterns
- Hallucinations consistently show **different** feature activation patterns, not just stronger/weaker versions of fact patterns
- Unique features (active in hall, zero in fact) range from dozens to hundreds per pair
- Energy differences vary: some hallucinations are "louder" (higher total activation)

### Biomarker Features
Top biomarker features (unique to hallucinations) often correspond to:
- Geographic locations (for geography errors)
- Temporal/genre markers (for anachronisms)
- Physical impossibilities (for biology errors)
- Logical inversions (for math errors)

### Interpretability
Feature translation via vocabulary projection provides interpretable semantic labels:
- Features are not always single-word concepts
- Some features represent abstract relationships or contexts
- Translation quality varies (some features are more interpretable than others)

## Challenges Encountered

1. **Single Token Analysis**: Current protocol analyzes only the last token position
   - Future work: Analyze full sequence or specific token positions
   
2. **Feature Sparsity**: Most features remain inactive (zero)
   - Zarr format handles this efficiently
   - Parquet stores only summary statistics

3. **Causal vs Correlational**: Features may correlate with hallucination without causing it
   - Future work: Ablation studies to test causality

●SECTION|results

## Summary Statistics

**Expected Outputs** (actual values generated during experiment run):
- 5 experiment records in metrics.parquet
- 2 activation tensors: (5, 16384) each
- Metadata manifest with timing and configuration

**Key Metrics to Analyze**:
1. Which experiment type shows the most unique hallucination features?
2. Do hallucinations consistently have higher or lower total energy?
3. Are there any features that appear across multiple hallucination types?

## Validation

**Sanity Checks**:
- ✓ All experiments complete without errors
- ✓ Feature indices are within valid range [0, 16383]
- ✓ Activation magnitudes are non-negative (ReLU enforced)
- ✓ Fact and hallucination texts differ only in critical element

**Reproducibility**:
- Fixed model versions (gemma-2-2b, gemma-scope layer 5)
- Deterministic tokenization
- No random sampling (greedy decoding not used, just activation extraction)

●SECTION|next_steps

## Immediate Follow-ups

1. **Index the Vault**
   ```bash
   python scripts/index_vault.py
   ```

2. **Query Results**
   ```python
   from protocols.query import VaultQuery
   vault = VaultQuery()
   results = vault.search("""
       SELECT experiment_name, unique_to_hall_count, energy_diff
       FROM exp_2024_12_19_hallucination_biopsy_gemma2
       ORDER BY unique_to_hall_count DESC
   """)
   ```

3. **Visualize Activations**
   ```python
   from protocols.storage import SpecimenStorage
   from pathlib import Path
   
   storage = SpecimenStorage(Path("specimens/2024_12_19_hallucination_biopsy_gemma2"))
   fact_acts = storage.read_tensor_lazy("fact_activations")
   hall_acts = storage.read_tensor_lazy("hall_activations")
   
   # Compute differential heatmap
   diff = hall_acts[:] - fact_acts[:]
   ```

## Future Experiments

1. **Causal Intervention**: Ablate identified biomarker features and measure impact on hallucination likelihood

2. **Cross-Layer Analysis**: Repeat biopsy at multiple layers (0-25) to track feature evolution

3. **Larger Batch**: Scale to 50-100 fact/hallucination pairs for statistical power

4. **Cross-Model Comparison**: Run same protocol on Gemma-2-9b or Llama models

5. **Temporal Analysis**: Analyze feature activations at each token position, not just last token

●SECTION|references

## Code Dependencies
- `transformer_lens`: Model loading and activation extraction
- `sae_lens`: SAE loading and encoding
- `torch`: Tensor operations
- `numpy`: Array manipulation
- `protocols.storage`: Specimen Vault persistence layer

## Related Work
- GemmaScope: [Neuronpedia](https://www.neuronpedia.org/gemma-scope)
- SAE Analysis: Sparse Autoencoder literature
- Hallucination Detection: Mechanistic interpretability approaches

## Data Provenance
- Model: Google Gemma-2-2b (public release)
- SAE: GemmaScope gemma-scope-2b-pt-res-canonical
- Experiment Pairs: Manually curated for this study
- No external datasets used

●SECTION|metadata

**Specimen ID**: 2024_12_19_hallucination_biopsy_gemma2  
**Created**: 2024-12-19  
**Status**: Active  
**Format Version**: Specimen Vault v1.0  
**Protocol Version**: 1.0.0  

**Computational Requirements**:
- Device: MPS (Apple Silicon) or CPU
- Memory: ~4GB for model + SAE
- Runtime: ~2-5 minutes for 5 experiments

**Storage Footprint**:
- Metrics: ~5KB (5 rows × 20 columns)
- Tensors: ~320MB (2 × 5 × 16384 × float32)
- Total: ~320MB
