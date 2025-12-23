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
<!-- Record experimental observations here after running -->

●SECTION|results
<!-- Summarize key findings here after running -->

**Key Metrics to Analyze:**
- Plot: X = noise rate, Y = semantic retention (separate lines for NL/VN)
- Calculate: "Breaking point" - noise rate where retention drops below 80%
- Compare: NL breaking point vs VN breaking point
- Visualize: "Viral chart" showing NL degradation vs VN resilience

●SECTION|next_steps
- Generate publication-ready figures (noise_robustness_plot.png)
- Create breakdown by complexity (noise_robustness_by_complexity.png)
- Analyze which noise types are most damaging
- Compare with Experiment H (Convergence Velocity) results
