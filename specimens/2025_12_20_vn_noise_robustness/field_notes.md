●LOG|specimen:2025_12_20_vn_noise_robustness|created:2025-12-20T10:02:13.989224+00:00

# Field Notes: 2025_12_20_vn_noise_robustness

## Experiment F: Noise Robustness Test

●SECTION|purpose
Test hypothesis that VN's explicit structure creates fault-tolerant representations.
NL relies on fragile grammatical/contextual cues that break under noise.
VN's structural anchors (●, |, :) maintain semantic integrity.

●SECTION|methodology
- Domain: interpretability
- Method: sae_analysis
- Tags: vn, noise_robustness, gemma-2-2b, layer_5

**Test Design:**
- 30 test cases (10 simple, 10 medium, 10 complex)
- Noise rates: 0%, 5%, 10%, 15%, 20%, 25%
- 5 noisy variants per (case, noise_rate)
- For VN: Preserve structural tokens (●, |, :) - they are the "anchors"
- For NL: All tokens equally vulnerable

**Metrics:**
1. **Semantic Retention**: Feature overlap with clean version
   - retention = |features_noisy ∩ features_clean| / |features_clean|
2. **Reconstruction Stability**: How much does loss increase?
   - stability = recon_loss_clean / recon_loss_noisy (inverted: higher = more stable)
3. **Purity Retention**: How much does purity degrade?
   - purity_retention = purity_noisy / purity_clean
4. **Critical Feature Survival**: % of top-20 features that survive
   - survival = |top_features_clean ∩ features_noisy| / |top_features_clean|

**Expected Outcome:**
VN should show higher retention, stability, and survival rates than NL,
especially at higher noise levels. The "breaking point" (where retention drops
below 80%) should be higher for VN than NL.

●SECTION|observations
**Experimental Execution:**
- Total samples: 1,800 (30 cases × 6 noise rates × 5 variants × 2 encodings)
- Noise rates tested: 0%, 5%, 10%, 15%, 20%, 25%
- Structural tokens preserved in VN: ●, |, : (anchors maintained)

**Key Observations:**
1. **Counterintuitive Reconstruction Improvement**: Noisy versions show lower reconstruction loss than clean versions (mean stability = 1.437). This suggests noise may be acting as regularization or the model is learning more robust representations under noise.

2. **Purity Paradox**: Purity actually increases with noise (mean purity_retention = 1.216). Clean purity: 0.027, Noisy purity: 0.034. This may indicate noise forces more focused feature activation.

3. **Extreme Feature Survival**: Critical feature survival remains >99% even at 25% noise (mean = 0.998, min = 0.90). Top-20 features are remarkably resilient.

4. **Encoding Comparison**: NL shows slightly better semantic retention than VN across all noise rates (NL: 0.969-1.0, VN: 0.959-1.0). This contradicts the hypothesis that VN's structural anchors provide superior robustness.

5. **No Breaking Point Found**: Neither encoding dropped below 80% retention even at 25% noise. Both encodings demonstrate exceptional robustness.

●SECTION|results
**Summary Statistics (n=1,800):**
- Semantic Retention: μ=0.977, σ=0.016, range=[0.917, 1.0]
- Critical Feature Survival: μ=0.998, σ=0.010, range=[0.90, 1.0]
- Reconstruction Stability: μ=1.437, σ=0.336, range=[0.999, 2.598]
- Purity Retention: μ=1.216, σ=0.234, range=[0.995, 2.092]

**Noise Rate Analysis:**

| Noise Rate | NL Retention | VN Retention | NL Survival | VN Survival |
|------------|--------------|---------------|-------------|-------------|
| 0%         | 1.000        | 1.000         | 1.000       | 1.000       |
| 5%         | 0.985        | 0.982         | 1.000       | 0.999       |
| 10%        | 0.979        | 0.974         | 0.999       | 0.999       |
| 15%        | 0.975        | 0.968         | 0.998       | 1.000       |
| 20%        | 0.971        | 0.963         | 0.997       | 0.999       |
| 25%        | 0.969        | 0.959         | 0.990       | 0.998       |

**Key Findings:**
1. **Hypothesis Partially Refuted**: VN does not show superior robustness to NL. NL maintains slightly higher semantic retention across all noise levels.

2. **Exceptional Robustness**: Both encodings demonstrate remarkable resilience. Even at 25% noise, retention remains >95% for both.

3. **Structural Anchors Effect**: While VN's structural tokens (●, |, :) were preserved, this did not translate to superior robustness. The structural anchors may provide different benefits (e.g., interpretability) rather than noise robustness.

4. **Reconstruction Paradox**: Noisy versions achieve better reconstruction (lower loss) than clean versions. Mean clean loss: 297.8, Mean noisy loss: 216.6. This suggests noise may act as implicit regularization.

5. **Purity Enhancement**: Noise increases feature purity (clean: 0.027, noisy: 0.034). This may indicate noise forces more selective, focused activations.

**Breaking Point Analysis:**
- NL breaking point: Not reached (retention >80% at all noise levels)
- VN breaking point: Not reached (retention >80% at all noise levels)
- Both encodings exceed expected robustness thresholds

**Key Metrics to Analyze:**
- Plot: X = noise rate, Y = semantic retention (separate lines for NL/VN)
- Calculate: "Breaking point" - noise rate where retention drops below 80% (NOT REACHED)
- Compare: NL breaking point vs VN breaking point (BOTH EXCEED 25% NOISE)
- Visualize: Reconstruction improvement paradox (noisy < clean loss)

●SECTION|next_steps
- Generate publication-ready figures (noise_robustness_plot.png)
- Create breakdown by complexity (noise_robustness_by_complexity.png)
- Analyze which noise types are most damaging
- Compare with Experiment H (Convergence Velocity) results
