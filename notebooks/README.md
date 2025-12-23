# Experiment Visualizations

This directory contains Jupyter notebooks for visualizing experiment results.

## Available Notebooks

### `visualize_results.ipynb`

Comprehensive visualization notebook for VN (Vector Native) vs NL (Natural Language) comparison experiments.

**Features:**
- Purity comparison scatter plots and distributions
- Category-wise analysis with grouped bar charts
- Complexity-wise analysis with violin plots
- Reconstruction loss visualizations (extended metrics)
- Token analysis (extended metrics)
- Per-token sparsity analysis (extended metrics)
- Top-K concentration plots (extended metrics)
- Feature overlap analysis
- Summary dashboard

## Usage

```bash
# From the experiments directory:
cd /Users/ariahan/Documents/ai-research/experiments

# Activate the virtual environment
source venv/bin/activate

# Launch Jupyter
jupyter notebook notebooks/visualize_results.ipynb
```

Or use JupyterLab:
```bash
jupyter lab notebooks/visualize_results.ipynb
```

## Supported Specimens

The notebook automatically detects available specimens and works with:

- `2025_12_22_vn_comprehensive_experiments` - Core purity metrics
- `2025_12_22_vn_extended_metrics` - Extended metrics including reconstruction loss, tokens, per-token sparsity

## Output

Figures are saved as PNG files in the notebook's working directory:
- `fig1_purity_comparison.png`
- `fig2_category_analysis.png`
- `fig3_complexity_analysis.png`
- `fig4_reconstruction_loss.png` (extended only)
- `fig5_token_analysis.png` (extended only)
- `fig6_per_token_sparsity.png` (extended only)
- `fig7_topk_concentration.png` (extended only)
- `fig8_feature_overlap.png`
- `fig9_summary_dashboard.png`

## Dependencies

All required packages are in `pyproject.toml`:
- `matplotlib` - Core plotting
- `seaborn` - Statistical visualizations
- `scipy` - Statistical tests
- `polars` - Data loading
- `jupyter` / `jupyterlab` - Notebook interface

