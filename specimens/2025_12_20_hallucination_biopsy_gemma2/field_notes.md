●LOG|specimen:2025_12_20_hallucination_biopsy_gemma2|created:2025-12-20T23:38:07.573883+00:00

# Field Notes: 2025_12_20_hallucination_biopsy_gemma2

●SECTION|purpose
Document observations, insights, and methodology for this experiment.

●SECTION|methodology
- Domain: interpretability
- Method: sae_analysis
- Tags: gemma-2-2b, layer_5, hallucination_detection, sae_diagnosis

●SECTION|observations
===========================================
QUICK STATS: 5 experiments
============================================================

Geography Teleportation:
  Unique features: 73
  Energy diff: +116.143
  Top feature: #9958 →  RB, RSD,  RCS

Geography Teleportation 2:
  Unique features: 40
  Energy diff: -136.787
  Top feature: #10496 →  York,  YORK, York

Historical Anachronism:
  Unique features: 102
  Energy diff: -578.276
  Top feature: #1059 → <bos>,  the, '

Biological Impossibility:
  Unique features: 78
  Energy diff: -54.940
  Top feature: #12485 →  wings,  Wings,  wing

Mathematical Inversion:
  Unique features: 22
  Energy diff: -32.006
  Top feature: #14143 → DeleteBehavior,  average,  dalamnya

============================================================
Averages:
  Unique features: 63.0
  Energy diff: -137.173
============================================================


●SECTION|results
The experiment identified 5 distinct hallucination signatures across 5 fact/hallucination pairs. The most common pattern was a "Geography Teleportation" effect where the model swapped locations without context, as seen in the top feature #9958 (RB, RSD, RCS). Other patterns included historical anachronisms (Shakespeare → Star Wars) and biological impossibilities (dogs → winged creatures). The mathematical inversion (five → two) was also present but less common.

●SECTION|next_steps
Will run the experiment with more test cases and more layers to see if the patterns hold.
