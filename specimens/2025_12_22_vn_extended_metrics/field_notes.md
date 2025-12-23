●LOG|specimen:2025_12_22_vn_extended_metrics|created:2025-12-22|executed:2025-12-22

# Field Notes: 2025_12_22_vn_extended_metrics

●SECTION|purpose

Extended experiment building on `vn_comprehensive_experiments` to address its limitations. Measures:

1. **SAE Reconstruction Loss** - The metric claimed in the paper but never measured in original experiment
2. **Token Count Analysis** - Actual tokenization comparison
3. **Feature Sparsity (L0, L1)** - Per-token sparsity measures
4. **Top-K Concentration** - Energy concentration in top features

●SECTION|execution

**Status:** ✓ COMPLETED 2025-12-22T20:30:41
**Run ID:** 20251222_203041
**Test Cases:** 75/75 processed successfully
**Data File:** runs/20251222_203041/metrics.parquet (75 rows × 38 columns)

●SECTION|critical_findings

### ⚠️ SURPRISING RESULT: VN Tokenizes to MORE Tokens

**VN prompts tokenize into 2.43x MORE tokens than NL on average.**

This is the OPPOSITE of what was expected. The VN syntax (with symbols like ●, |, :, etc.) creates more tokens, not fewer.

| Complexity | NL Tokens | VN Tokens | Token Ratio |
|------------|-----------|-----------|-------------|
| Simple | 14.3 | 30.6 | 2.17x |
| Medium | 33.7 | 77.6 | 2.37x |
| Complex | 79.9 | 222.4 | 2.71x |

**Implication:** VN is NOT more "token efficient" in the sense of fewer tokens. However...

### ✓ VN Has Lower Per-Token Reconstruction Loss

Despite using more tokens, each VN token has significantly lower reconstruction loss:

| Metric | NL | VN | Improvement |
|--------|----|----|-------------|
| Recon Loss Mean (per token) | 498,447 | 244,340 | **-51.0%** |
| Recon Loss Std | 334,381 | 180,848 | -45.9% |
| t-statistic | — | — | t=11.16 |
| p-value | — | — | p=1.59e-17 |

**Interpretation:** The SAE can reconstruct VN token activations with half the error. VN activations align better with the model's learned feature dictionary.

### ✓ VN Has Fewer Features Per Token (Sparser)

Per-token sparsity analysis reveals VN IS sparser at the per-token level:

| Metric | NL | VN | Change |
|--------|----|----|--------|
| L0 Mean (features/token) | 705.5 | 598.8 | **-15.1%** |
| L0 Fraction (of 16k) | 4.3% | 3.7% | -0.6pp |

**Interpretation:** Each VN token activates 15% fewer features on average. The previous experiment showed VN activating MORE total features because VN has more tokens—but per token, VN is sparser.

### ✓ VN Has Higher Top-K Concentration

VN activations are more concentrated in top features:

| Top-K | NL Concentration | VN Concentration | Δ |
|-------|------------------|------------------|---|
| Top-10 | 2.66% | 2.97% | +11.7% |
| Top-50 | 5.61% | 6.91% | +23.2% |
| Top-100 | 8.13% | 10.19% | +25.3% |
| Top-500 | 20.69% | 25.42% | +22.9% |

**Interpretation:** More of VN's activation energy is concentrated in fewer features. This suggests cleaner, more focused representations.

●SECTION|reconciling_with_original_experiment

The original `vn_comprehensive_experiments` showed:
- VN activates MORE total features (9,554 vs 9,163)
- VN achieves HIGHER purity

This new experiment explains WHY:
1. VN has MORE tokens (2.4x), so total features = per_token × n_tokens
2. Per token, VN is actually SPARSER (599 vs 706 features)
3. Per token, VN has LOWER reconstruction loss (51% improvement)
4. Per token, VN is more CONCENTRATED in top features

**The original finding was a Simpson's Paradox situation.** VN appeared to activate more features because it has more tokens, not because each token activates more.

●SECTION|verified_results

### Overall Statistics (n=75)

| Metric | Value |
|--------|-------|
| Reconstruction Loss Improvement | **51.5% ± 14.5%** |
| Purity Improvement | 43.9% ± 30.0% |
| Token Ratio (VN/NL) | 2.43 |
| L0 Improvement (per token) | -15.1% |
| Top-100 Concentration Improvement | +25.3% |

### By Complexity

| Complexity | n | Recon Improvement | Purity Improvement | Token Ratio |
|------------|---|-------------------|-------------------|-------------|
| Simple | 21 | +49.3% | +26.4% | 2.17 |
| Medium | 28 | +50.4% | +51.0% | 2.37 |
| Complex | 26 | +54.3% | +50.4% | 2.71 |

### Statistical Significance

| Test | Statistic | Result |
|------|-----------|--------|
| Paired t-test (recon) | t = 11.16 | p = 1.59e-17 |
| Effect size | — | Very large |

●SECTION|implications_for_paper

**Can Claim:**
1. ✓ VN has 51% lower SAE reconstruction loss per token (verified, p<0.0001)
2. ✓ VN activations are sparser per token (15% fewer features)
3. ✓ VN is more concentrated in top features (25% higher top-100 concentration)
4. ✓ VN has higher purity (44% improvement)

**CANNOT Claim:**
1. ❌ VN is "more token efficient" (VN uses 2.4x MORE tokens)
2. ❌ VN activates fewer total features (VN activates more due to more tokens)
3. ❌ VN is "more compact" at the input level

**Nuanced Framing:**
"While VN syntax tokenizes into more tokens due to special characters, each token produces activations that better align with the model's learned representations (51% lower reconstruction loss) and are more focused (15% sparser per token, 25% higher concentration)."

●SECTION|data_schema

Columns in metrics.parquet (38 total):

**Identifiers:** case_id, category, complexity, description
**Tokens:** nl_tokens, vn_tokens, token_ratio
**Purity:** nl_purity, vn_purity, purity_improvement_pct
**Reconstruction:** nl_recon_loss_mean/std, vn_recon_loss_mean/std, recon_improvement_pct, nl_relative_error, vn_relative_error
**Sparsity:** nl_l0_mean, vn_l0_mean, nl_l0_fraction, vn_l0_fraction, nl_l1_mean, vn_l1_mean
**Concentration:** nl_topk_{10,50,100,500}, vn_topk_{10,50,100,500}
**Token-Normalized:** nl_features_per_token, vn_features_per_token, nl_intensity_per_token, vn_intensity_per_token
**Counts:** nl_active_features, vn_active_features, run_id

●SECTION|next_steps

1. ✓ Reconstruction loss measured and verified
2. ✓ Token count analysis completed
3. ✓ Per-token sparsity analysis completed
4. **TODO:** Update paper with corrected claims
5. **TODO:** Run on multiple layers (not just layer 5)
6. **TODO:** Test on different models for generalization
7. **TODO:** Correlate with downstream task performance
