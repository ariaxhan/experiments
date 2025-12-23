●LOG|specimen:2025_12_22_vn_convergence_velocity|created:2025-12-22T10:02:13.989224+00:00

# Field Notes: 2025_12_22_vn_convergence_velocity

## Experiment H: Convergence Velocity Test

●SECTION|purpose
Test hypothesis that VN reaches stable task representations in earlier layers.
The model "knows what to do" faster with VN because there's less 
linguistic ambiguity to resolve.

●SECTION|methodology
- Domain: interpretability
- Method: layer_analysis
- Tags: vn, convergence, velocity, gemma-2-2b, multi_layer

**Test Design:**
- 30 test cases (10 simple, 10 medium, 10 complex)
- Extract residual stream activations at layers: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25]
- Use last token position (where prediction happens)
- Final layer: 25 (for convergence comparison)

**Metrics:**
1. **Final-Layer Similarity**: cosine_sim(activation[L], activation[final])
   - Measures "how close is layer L to the final answer?"
   - Tracked across all sampled layers
2. **Convergence Velocity**: First layer where similarity > 0.9
   - Lower = faster = better
   - velocity_NL vs velocity_VN
3. **Representation Stability**: cosine_sim(activation[L], activation[L+1])
   - High stability = representation has "settled"
   - Find first layer where stability > 0.95 for 3 consecutive layers
4. **Cross-Encoding Convergence**: cosine_sim(NL_activation[L], VN_activation[L])
   - Do NL and VN converge to the same final representation?
   - Or do they stay in different regions of activation space?
5. **Task-Concept Alignment**: cosine_sim(activation[L], task_vector)
   - How quickly does each encoding align with its task vector?
   - Task vectors computed as average activations per category

**Expected Outcome:**
VN should converge earlier (lower layer number) than NL.
VN should show higher final-layer similarity in earlier layers.
VN should stabilize faster (reach stability point earlier).
Cross-encoding convergence should show whether NL and VN converge to same or different representations.

●SECTION|visualizations
1. **Convergence Curves**: X = layer, Y = final-layer similarity
   - Separate lines for NL vs VN, with confidence bands
   - Shows when each encoding "converges" to final representation
2. **Velocity Histogram**: Distribution of "convergence layer" for NL vs VN
   - Shows distribution of convergence points
   - Lower = faster convergence
3. **Trajectory Plot**: PCA of activations across layers
   - Show NL and VN paths through representation space
   - One plot per test case
4. **Cross-Encoding Convergence**: NL-VN similarity across layers
   - Shows whether encodings converge to same or different representations

●SECTION|observations
<!-- Record experimental observations here after running -->

●SECTION|results
<!-- Summarize key findings here after running -->

**Key Metrics to Analyze:**
- Mean convergence velocity: NL vs VN
- Median convergence velocity: NL vs VN
- Percentage of cases that converge: NL vs VN
- VN advantage: How many layers earlier does VN converge?
- Cross-encoding convergence: Do NL and VN converge to same representation?
- Task alignment: Which encoding aligns faster with task vectors?

●SECTION|next_steps
- Analyze convergence patterns by complexity level
- Compare with Experiment F (Noise Robustness) results
- Investigate cases where VN doesn't converge faster
- Analyze task-specific convergence patterns
- Create publication-ready figures with statistical significance tests

