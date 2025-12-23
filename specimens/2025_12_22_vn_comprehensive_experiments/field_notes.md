●LOG|specimen:2025_12_22_vn_comprehensive_experiments|created:2025-12-22|updated:2025-12-22

# Field Notes: 2025_12_22_vn_comprehensive_experiments

●SECTION|purpose
Comprehensive validation of Vector-Native (VN) syntax effectiveness across diverse domains and complexity levels using Sparse Autoencoder (SAE) analysis. This experiment runs 75 test cases comparing Natural Language (NL) vs Vector-Native syntax to measure spectral purity improvements and feature activation patterns.

●SECTION|methodology

**Model & SAE Configuration:**
- Model: Gemma-2-2B (via TransformerLens)
- SAE: Gemma-Scope layer_5/width_16k/canonical
- Hook point: `blocks.5.hook_resid_post` (residual stream post layer 5)
- SAE dictionary size: 16,384 features

**Test Cases:**
- Total: 75 matched NL/VN pairs
- Categories: 5 (data_analysis:20, development:20, machine_learning:15, content:10, operations:10)
- Complexities: simple:21, medium:28, complex:26

**Processing Pipeline (per test case):**
1. Tokenize NL/VN text via model.to_tokens()
2. Forward pass with cache via run_with_cache()
3. Extract activations from blocks.5.hook_resid_post [batch, tokens, d_model]
4. Flatten to [batch*tokens, d_model]
5. Encode via SAE: sae.encode() → [batch*tokens, 16384]
6. Threshold active features (|activation| > 1e-6)
7. Aggregate by feature index: sum intensities across all tokens
8. Create Spectrum object (wavelengths=feature_indices, intensities=summed_magnitudes)

**Purity Calculation (InterferenceEngine):**
```
P = 0.6 × entropy_purity + 0.4 × snr_purity

entropy_purity = 1 - (entropy / max_entropy)
  where entropy = -Σ(p_i × log(p_i)), p_i = intensity_i / Σintensities

snr_purity = min(1.0, snr / 100)
  where snr = signal_power / noise_power
  signal = features with intensity >= 10% of max intensity
```

●SECTION|verified_results

**Execution:** 2025-12-22, 2 runs, identical results (deterministic)
**Data Source:** runs/20251222_201819/metrics.parquet (75 rows, 18 columns)

### Overall Statistics

| Metric | NL | VN | Change |
|--------|----|----|--------|
| Mean Purity | 0.0221 ± 0.0039 | 0.0314 ± 0.0122 | +42.0% |
| Min Purity | 0.0180 | 0.0189 | — |
| Max Purity | 0.0325 | 0.0618 | — |
| Mean Features | 9,163 | 9,554 | **+4.3%** (VN uses more) |
| Mean Overlap | 8,875 features | 90.3% overlap | — |

### By Complexity

| Complexity | n | NL Purity | VN Purity | Improvement | NL Features | VN Features | Overlap |
|------------|---|-----------|-----------|-------------|-------------|-------------|---------|
| Simple | 21 | 0.0186 | 0.0205 | +10.4% | 8,798 | 9,010 | 93.5% |
| Medium | 28 | 0.0205 | 0.0270 | +31.9% | 9,094 | 9,460 | 90.4% |
| Complex | 26 | 0.0267 | 0.0450 | +66.9% | 9,532 | 10,094 | 87.7% |

### By Category

| Category | n | Mean Improvement | Std Dev |
|----------|---|------------------|---------|
| data_analysis | 20 | +23.6% | 17.1% |
| machine_learning | 15 | +38.5% | 28.7% |
| development | 20 | +42.4% | 27.9% |
| content | 10 | +47.3% | 38.1% |
| operations | 10 | +47.9% | 35.8% |

### Statistical Significance

| Statistic | Value |
|-----------|-------|
| Paired t-test | t = 9.4461, p = 2.40e-14 |
| Cohen's d | 1.09 ("very large" effect) |
| 95% CI | [31.29%, 44.71%] |
| Mean improvement | 38.0% |
| Median improvement | 34.1% |
| Std improvement | 29.2% |
| IQR | [12.8%, 53.4%] |

### Top 10 Performers

| Rank | Case ID | Category | Complexity | Improvement |
|------|---------|----------|------------|-------------|
| 1 | content_009 | content | complex | +116.3% |
| 2 | dev_020 | development | complex | +102.8% |
| 3 | content_003 | content | complex | +100.7% |
| 4 | ops_009 | operations | complex | +96.8% |
| 5 | ops_003 | operations | complex | +94.9% |
| 6 | dev_018 | development | complex | +91.4% |
| 7 | ml_015 | machine_learning | complex | +84.3% |
| 8 | ops_006 | operations | complex | +84.0% |
| 9 | ml_012 | machine_learning | complex | +83.8% |
| 10 | dev_012 | development | complex | +80.1% |

### Negative/Low Performers

| Case ID | Category | Complexity | Improvement |
|---------|----------|------------|-------------|
| data_analysis_002 | data_analysis | medium | **-0.63%** |
| data_analysis_017 | data_analysis | simple | +1.4% |
| data_analysis_001 | data_analysis | simple | +1.9% |
| data_analysis_006 | data_analysis | medium | +3.6% |
| ml_001 | machine_learning | simple | +4.7% |

Only 1 case showed negative improvement (VN performed worse than NL).

### Feature Count Analysis

- VN uses MORE features than NL: **74 / 75 cases** (98.7%)
- VN uses FEWER features than NL: 1 / 75 cases (1.3%)
- Average feature increase: +391 features (+4.3%)

This is COUNTER to the hypothesis that VN is "sparser." VN actually activates more features but with higher signal concentration.

●SECTION|critical_observations

**1. VN is NOT Sparser—It's Cleaner**

The experiment contradicts any hypothesis that VN produces fewer active features. VN consistently activates MORE features (74/75 cases). However, VN achieves higher purity, meaning:
- The additional features have higher signal-to-noise ratio
- Activation energy is more concentrated in meaningful features
- The "noise floor" of weak activations is relatively lower

**2. Complexity Scaling is Non-Linear**

The 6.4x improvement gap between simple (+10.4%) and complex (+66.9%) tasks suggests VN's value compounds with task complexity. This aligns with the hypothesis that structured syntax "absorbs" complexity that would otherwise manifest as semantic ambiguity.

**3. Data Analysis Underperforms**

Data analysis tasks show the weakest VN advantage (+23.6% vs ~47% for content/operations). Possible explanations:
- Analytical queries are already semi-structured in NL ("calculate X from Y")
- The VN syntax for queries (●query|...) may not provide sufficient structural advantage
- Domain-specific investigation needed

**4. The Negative Case**

`data_analysis_002` is the only case where VN underperformed:
- NL: "Please analyze the Q4 2024 sales data and break it down by region, showing revenue and profit margins for each region"
- VN: `●analyze|dataset:Q4_2024_sales|groupby:region|metrics:revenue,profit_margin`
- NL Purity: 0.0200, VN Purity: 0.0198 (-0.63%)

The VN version is notably shorter but loses context. "Please analyze" in NL may activate beneficial meta-cognitive features that the terse VN version misses.

**5. High Feature Overlap**

90.3% average feature overlap between NL and VN indicates:
- Both syntaxes activate largely the same semantic concepts
- VN doesn't fundamentally change WHAT is represented, only HOW
- The improvement is in signal quality, not semantic shift

●SECTION|methodological_concerns

1. **Token Length Confound:** Intensities are summed across all tokens. Longer inputs = more tokens = more opportunity for feature activation. VN prompts are typically shorter in characters but may tokenize differently.

2. **No Token Normalization:** We did not normalize by token count. A proper comparison might use mean intensity per token.

3. **Single Model/SAE:** Results are from Gemma-2-2B layer 5 only. Generalization to other models/layers is unknown.

4. **Test Case Construction Bias:** Both NL and VN versions were manually created. The NL versions may be artificially verbose.

5. **Purity vs Performance:** We measured purity, not downstream task performance. High purity may or may not correlate with better model outputs.

●SECTION|what_was_not_measured

- ❌ SAE Reconstruction Loss (L_rec = ||x - x_hat||²)
- ❌ Downstream task performance (answer quality)
- ❌ Token-normalized metrics
- ❌ Multi-layer analysis
- ❌ Hallucination rate correlation

●SECTION|data_files

| File | Path | Contents |
|------|------|----------|
| Metrics | runs/20251222_201819/metrics.parquet | 75 rows × 18 columns |
| Manifest | runs/20251222_201819/manifest.msgpack | Experiment metadata |
| Legacy Strata | strata/metrics.parquet | Same data, legacy location |

Columns in metrics.parquet:
- case_id, category, complexity, description
- nl_text, vn_text, expected_reduction
- nl_purity, vn_purity, purity_improvement_pct
- nl_features, vn_features, feature_reduction_pct
- overlap_count, overlap_pct
- vn_unique_count, nl_unique_count, run_id

●SECTION|next_steps

1. **Token Normalization:** Re-run with per-token intensity averaging
2. **Reconstruction Loss:** Measure actual L_rec = ||x - sae.decode(sae.encode(x))||²
3. **Multi-Layer Analysis:** Test layers 3, 5, 8, 12, 18 to see if effect varies by depth
4. **Downstream Validation:** Measure if high-purity prompts produce better outputs
5. **Negative Case Analysis:** Deep dive into data_analysis_002 to understand failure mode
6. **Model Generalization:** Test on Gemma-7B, Llama-2, etc.
