# Universal Results Viewer

A comprehensive tool for viewing and analyzing experiment results across all specimen types in the Specimen Vault.

## Features

- **Automatic Detection**: Automatically detects experiment type and metric structure
- **Comprehensive Statistics**: Calculates mean, std, percentiles, and more for all metrics
- **Experiment-Specific Analysis**: Provides specialized insights for different experiment types:
  - Noise Robustness: Breaking points, retention curves
  - VN Comparison: Purity improvements, feature reductions
  - Hallucination Analysis: Feature differences, energy metrics
- **Manifest Integration**: Shows experiment metadata (domain, method, tags, timestamps)
- **JSON Export**: Export full analysis to JSON for further processing
- **Comparison Mode**: Compare multiple specimens side-by-side

## Usage

### View Specific Specimen

```bash
# From repository root
python scripts/view_results.py specimens/2024_12_20_vn_noise_robustness

# Or use relative path
python scripts/view_results.py specimens/2024_12_20_vn_comprehensive_experiments
```

### View All Specimens

```bash
python scripts/view_results.py --all
```

### Compare Specimens

```bash
python scripts/view_results.py --compare \
    specimens/2024_12_20_vn_noise_robustness \
    specimens/2024_12_20_vn_comprehensive_experiments
```

### Export to JSON

```bash
python scripts/view_results.py specimens/2024_12_20_vn_noise_robustness \
    --json analysis.json
```

### Quick Stats (Backward Compatible)

For backward compatibility with existing `quick_stats.py` scripts:

```bash
# From specimen directory
cd specimens/2024_12_20_vn_noise_robustness
python ../../scripts/quick_stats.py

# Or specify path
python scripts/quick_stats.py specimens/2024_12_20_vn_noise_robustness
```

## Output Sections

The viewer provides several sections of analysis:

1. **EXPERIMENT METADATA**: Specimen ID, domain, method, tags, timestamps
2. **METRIC OVERVIEW**: Experiment type, record count, detected metrics
3. **STATISTICS**: Comprehensive stats (mean, std, percentiles) for key metrics
4. **EXPERIMENT-SPECIFIC ANALYSIS**: 
   - Noise Robustness: Breaking points, retention by noise rate
   - VN Comparison: Overall comparison, by complexity, top improvements

## Supported Experiment Types

- **noise_robustness**: Experiments with noise injection and encoding comparison
- **vn_comparison**: NL vs VN purity and feature comparisons
- **hallucination_analysis**: Feature differences and energy metrics
- **test_case_analysis**: Test case-based experiments with categories

## Integration

To add a `quick_stats.py` to any specimen directory for convenience:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from scripts.view_results import view_specimen
view_specimen(Path(__file__).parent)
```

## Examples

### Noise Robustness Analysis

```bash
$ python scripts/view_results.py specimens/2024_12_20_vn_noise_robustness

================================================================================
 SPECIMEN: 2024_12_20_vn_noise_robustness
================================================================================

================================================================================
 EXPERIMENT METADATA
================================================================================

Specimen ID: 2024_12_20_vn_noise_robustness
Domain: interpretability
Method: sae_analysis
Tags: vn, noise_robustness, gemma-2-2b, layer_5

================================================================================
 METRIC OVERVIEW
================================================================================

Experiment Type: noise_robustness
Total Records: 1,800
Numeric Metrics: 15
Categorical Columns: 4

Key Metrics Detected:
  • semantic_retention
  • reconstruction_stability
  • purity_retention
  • critical_feature_survival
  • clean_purity
  • noisy_purity

================================================================================
 EXPERIMENT-SPECIFIC ANALYSIS
================================================================================

Breaking Points (Retention < 80%):
  NL: 15% noise
  VN: 25% noise

Retention by Noise Rate:
  Noise Rate   NL Retention    VN Retention
  ------------------------------------------
      0%      1.000         1.000
      5%      0.950         0.980
     10%      0.850         0.920
     15%      0.750         0.880
     20%      0.650         0.820
     25%      0.550         0.750
```

## JSON Export Format

The JSON export includes:

```json
{
  "specimen_id": "2024_12_20_vn_noise_robustness",
  "classification": {
    "experiment_type": "noise_robustness",
    "key_metrics": [...],
    ...
  },
  "statistics": {
    "semantic_retention": {
      "count": 1800,
      "mean": 0.85,
      "std": 0.12,
      ...
    },
    ...
  },
  "noise_analysis": {
    "breaking_points": {...},
    "by_noise_rate": {...}
  }
}
```



