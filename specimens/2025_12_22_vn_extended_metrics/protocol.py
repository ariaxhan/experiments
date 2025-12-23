"""●COMPONENT|Ψ:vn_extended_metrics_protocol|Ω:comprehensive_sae_analysis_with_reconstruction_loss

Extended Vector-Native Experiments Runner

This experiment extends vn_comprehensive_experiments by measuring:
1. Spectral Purity (entropy + SNR based) - as before
2. SAE Reconstruction Loss - L_rec = ||x - x_hat||²
3. Token-Normalized Metrics - per-token averages to control for input length
4. Feature Sparsity Metrics - L0, L1 norms of feature activations
5. Top-K Feature Concentration - what % of activation energy is in top-k features

Reuses components from engines/universal_spectroscopy and tests/vn_test_cases.

Dependencies: transformer_lens, sae_lens, torch, polars
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
import numpy as np

from protocols.storage import SpecimenStorage
from tests.vn_test_cases import (
    TEST_CASES,
    get_test_cases_by_category,
    get_test_case_stats,
    get_all_categories
)
from engines.universal_spectroscopy import UniversalSpectroscopyEngine, get_device
from engines.universal_spectroscopy.spectrum import Spectrum


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

CONFIG = {
    "run_all_test_cases": True,
    "categories_to_test": None,  # None = all
    "decode_features": False,  # Skip feature decoding for speed
    "top_k_concentration": [10, 50, 100, 500],  # k values for top-k concentration
}


# ============================================================================
# SECTION 2: EXTENDED METRICS
# ============================================================================

def calculate_reconstruction_loss(
    sae,
    activations_flat: torch.Tensor
) -> Dict[str, float]:
    """●METHOD|input:sae_tensor|output:dict|operation:compute_sae_reconstruction_metrics
    
    Compute SAE reconstruction loss and related metrics.
    
    L_rec = ||x - sae.decode(sae.encode(x))||²
    
    Args:
        sae: Loaded SAE object from sae_lens
        activations_flat: Flattened activations [n_tokens, d_model]
        
    Returns:
        Dict with reconstruction metrics
    """
    with torch.no_grad():
        # Encode and decode
        encoded = sae.encode(activations_flat)
        reconstructed = sae.decode(encoded)
        
        # Per-token reconstruction loss (sum over hidden dim)
        per_token_loss = (activations_flat - reconstructed).pow(2).sum(dim=-1)
        
        # Aggregate
        total_loss = per_token_loss.sum().item()
        mean_loss = per_token_loss.mean().item()
        std_loss = per_token_loss.std().item() if len(per_token_loss) > 1 else 0.0
        max_loss = per_token_loss.max().item()
        min_loss = per_token_loss.min().item()
        
        # Relative reconstruction error (normalized by input magnitude)
        input_norm = activations_flat.pow(2).sum(dim=-1).mean().item()
        relative_error = mean_loss / (input_norm + 1e-10)
        
        return {
            "recon_loss_total": total_loss,
            "recon_loss_mean": mean_loss,
            "recon_loss_std": std_loss,
            "recon_loss_max": max_loss,
            "recon_loss_min": min_loss,
            "recon_relative_error": relative_error,
            "input_norm_mean": input_norm,
        }


def calculate_sparsity_metrics(
    feature_activations: torch.Tensor
) -> Dict[str, float]:
    """●METHOD|input:tensor|output:dict|operation:compute_feature_sparsity_metrics
    
    Compute sparsity metrics for feature activations.
    
    Args:
        feature_activations: [n_tokens, n_features]
        
    Returns:
        Dict with sparsity metrics
    """
    with torch.no_grad():
        # L0 norm per token (count of non-zero features)
        active_mask = feature_activations.abs() > 1e-6
        l0_per_token = active_mask.sum(dim=-1).float()
        
        # L1 norm per token (sum of absolute activations)
        l1_per_token = feature_activations.abs().sum(dim=-1)
        
        # Aggregate
        n_features = feature_activations.shape[-1]
        
        return {
            "l0_mean": l0_per_token.mean().item(),
            "l0_std": l0_per_token.std().item() if len(l0_per_token) > 1 else 0.0,
            "l0_max": l0_per_token.max().item(),
            "l0_fraction": (l0_per_token.mean() / n_features).item(),  # fraction of features active
            "l1_mean": l1_per_token.mean().item(),
            "l1_std": l1_per_token.std().item() if len(l1_per_token) > 1 else 0.0,
        }


def calculate_topk_concentration(
    feature_activations: torch.Tensor,
    k_values: List[int]
) -> Dict[str, float]:
    """●METHOD|input:tensor_list|output:dict|operation:compute_topk_energy_concentration
    
    Compute what fraction of total activation energy is in top-k features.
    
    Args:
        feature_activations: [n_tokens, n_features]
        k_values: List of k values to compute
        
    Returns:
        Dict with top-k concentration metrics
    """
    with torch.no_grad():
        # Aggregate across tokens first (sum intensities per feature)
        aggregated = feature_activations.abs().sum(dim=0)  # [n_features]
        total_energy = aggregated.sum().item()
        
        if total_energy < 1e-10:
            return {f"topk_{k}_concentration": 0.0 for k in k_values}
        
        # Sort features by intensity
        sorted_intensities, _ = torch.sort(aggregated, descending=True)
        
        results = {}
        for k in k_values:
            if k >= len(sorted_intensities):
                topk_energy = total_energy
            else:
                topk_energy = sorted_intensities[:k].sum().item()
            
            results[f"topk_{k}_concentration"] = topk_energy / total_energy
        
        return results


def calculate_token_normalized_purity(
    spectrum: Spectrum,
    n_tokens: int
) -> Dict[str, float]:
    """●METHOD|input:spectrum_int|output:dict|operation:compute_token_normalized_metrics
    
    Compute per-token normalized versions of metrics.
    
    Args:
        spectrum: Spectrum object
        n_tokens: Number of tokens in the input
        
    Returns:
        Dict with token-normalized metrics
    """
    if n_tokens <= 0:
        n_tokens = 1
    
    intensities = spectrum.intensities
    
    return {
        "n_tokens": n_tokens,
        "features_per_token": len(spectrum) / n_tokens,
        "intensity_per_token": intensities.sum().item() / n_tokens,
        "max_intensity": intensities.max().item() if len(intensities) > 0 else 0.0,
        "mean_intensity_per_feature": intensities.mean().item() if len(intensities) > 0 else 0.0,
    }


# ============================================================================
# SECTION 3: EXTENDED PROCESSING
# ============================================================================

def process_text_extended(
    text: str,
    engine: UniversalSpectroscopyEngine,
    layer: int = 5
) -> Tuple[Spectrum, Dict[str, Any]]:
    """●METHOD|input:str_engine_int|output:tuple|operation:process_text_with_extended_metrics
    
    Process text and compute all extended metrics.
    
    Args:
        text: Input text
        engine: UniversalSpectroscopyEngine instance
        layer: Layer to extract from
        
    Returns:
        Tuple of (Spectrum, extended_metrics_dict)
    """
    # Tokenize to get token count
    tokens = engine.model.to_tokens(text)
    n_tokens = tokens.shape[1]
    tokens = tokens.to(engine.device)
    
    # Get activations
    with torch.no_grad():
        _, cache = engine.model.run_with_cache(tokens, return_type=None)
        activation_key = f"blocks.{layer}.hook_resid_post"
        activations = cache[activation_key]  # [batch, tokens, d_model]
    
    # Flatten for SAE processing
    activations_flat = activations.view(-1, activations.shape[-1])
    
    # Get SAE
    sae = engine.sae_adapter.sae
    
    # Compute reconstruction loss
    recon_metrics = calculate_reconstruction_loss(sae, activations_flat)
    
    # Get feature activations for sparsity metrics
    with torch.no_grad():
        feature_activations = sae.encode(activations_flat)
    
    # Sparsity metrics
    sparsity_metrics = calculate_sparsity_metrics(feature_activations)
    
    # Top-k concentration
    topk_metrics = calculate_topk_concentration(feature_activations, CONFIG["top_k_concentration"])
    
    # Process through engine to get spectrum
    spectrum = engine.process(text, layer=layer)
    
    # Token-normalized metrics
    token_metrics = calculate_token_normalized_purity(spectrum, n_tokens)
    
    # Standard purity (for comparison)
    purity = engine.calculate_purity(spectrum)
    
    # Combine all metrics
    extended_metrics = {
        **recon_metrics,
        **sparsity_metrics,
        **topk_metrics,
        **token_metrics,
        "purity": purity,
        "n_active_features": len(spectrum),
    }
    
    return spectrum, extended_metrics


# ============================================================================
# SECTION 4: MAIN EXPERIMENT
# ============================================================================

def run_experiment() -> None:
    """●METHOD|input:None|output:None|operation:execute_vn_extended_metrics_experiment"""
    
    experiment_start_time = datetime.now().isoformat()
    
    # Initialize storage
    specimen_path = Path(__file__).parent
    storage = SpecimenStorage(specimen_path)
    
    print(f"●PROCESS|operation:vn_extended_metrics|phase:starting")
    print(f"  Specimen: {specimen_path.name}")
    print(f"  Run ID: {storage.run_id}")
    
    print("\n" + "="*80)
    print("EXTENDED VECTOR-NATIVE METRICS EXPERIMENT")
    print("Measuring: Reconstruction Loss, Sparsity, Top-K Concentration, Token-Normalized Purity")
    print("="*80)
    
    # Initialize device and engine
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Print test case statistics
    stats = get_test_case_stats()
    print(f"\nTest Cases Library Statistics:")
    print(f"  Total available: {stats['total']}")
    print(f"  Categories: {len(stats['by_category'])}")
    print(f"  Complexities: {', '.join(f'{k}:{v}' for k, v in stats['by_complexity'].items())}")
    
    # Initialize engine
    print("\n" + "="*80)
    print("INITIALIZING UNIVERSAL SPECTROSCOPY ENGINE")
    print("="*80)
    engine = UniversalSpectroscopyEngine(device=device)
    
    # Load model
    print("\n>> Loading Gemma-2-2B model...")
    engine.load_model("gemma-2-2b")
    print("✓ Model loaded")
    
    # Load SAE
    print("\n>> Loading SAE from Gemma-Scope...")
    engine.load_sae(
        model_name="gemma-2-2b",
        layer=5,
        release="gemma-scope-2b-pt-res-canonical",
        sae_id="layer_5/width_16k/canonical"
    )
    print("✓ SAE loaded")
    
    # Select test cases
    print("\n" + "="*80)
    print("SELECTING TEST CASES")
    print("="*80)
    
    if CONFIG["run_all_test_cases"]:
        if CONFIG["categories_to_test"] is None:
            selected_cases = [(k, v) for k, v in TEST_CASES.items()]
        else:
            selected_cases = [(k, v) for k, v in TEST_CASES.items() 
                            if v['category'] in CONFIG["categories_to_test"]]
    
    print(f"\nRunning {len(selected_cases)} test cases")
    
    # Run experiments
    print("\n" + "="*80)
    print("RUNNING EXPERIMENTS")
    print("="*80)
    
    results = []
    processed = 0
    total = len(selected_cases)
    
    for case_id, case_data in selected_cases:
        processed += 1
        print(f"\n[{processed}/{total}] Processing: {case_id}")
        print(f"  Category: {case_data['category']} | Complexity: {case_data['complexity']}")
        
        try:
            # Process NL version
            nl_spectrum, nl_metrics = process_text_extended(case_data['nl'], engine)
            
            # Process VN version
            vn_spectrum, vn_metrics = process_text_extended(case_data['vn'], engine)
            
            # Calculate improvements/differences
            purity_improvement = (vn_metrics['purity'] - nl_metrics['purity']) / nl_metrics['purity'] * 100 if nl_metrics['purity'] > 0 else 0
            recon_improvement = (nl_metrics['recon_loss_mean'] - vn_metrics['recon_loss_mean']) / nl_metrics['recon_loss_mean'] * 100 if nl_metrics['recon_loss_mean'] > 0 else 0
            
            print(f"  NL: purity={nl_metrics['purity']:.4f}, recon_loss={nl_metrics['recon_loss_mean']:.2f}, tokens={nl_metrics['n_tokens']}")
            print(f"  VN: purity={vn_metrics['purity']:.4f}, recon_loss={vn_metrics['recon_loss_mean']:.2f}, tokens={vn_metrics['n_tokens']}")
            print(f"  Δ Purity: {purity_improvement:+.1f}%, Δ Recon: {recon_improvement:+.1f}%")
            
            # Store result
            result = {
                "case_id": case_id,
                "category": case_data['category'],
                "complexity": case_data['complexity'],
                "description": case_data['description'],
                
                # Token counts
                "nl_tokens": nl_metrics['n_tokens'],
                "vn_tokens": vn_metrics['n_tokens'],
                "token_ratio": vn_metrics['n_tokens'] / nl_metrics['n_tokens'] if nl_metrics['n_tokens'] > 0 else 0,
                
                # Purity
                "nl_purity": nl_metrics['purity'],
                "vn_purity": vn_metrics['purity'],
                "purity_improvement_pct": purity_improvement,
                
                # Reconstruction Loss
                "nl_recon_loss_mean": nl_metrics['recon_loss_mean'],
                "vn_recon_loss_mean": vn_metrics['recon_loss_mean'],
                "recon_improvement_pct": recon_improvement,
                "nl_recon_loss_std": nl_metrics['recon_loss_std'],
                "vn_recon_loss_std": vn_metrics['recon_loss_std'],
                "nl_relative_error": nl_metrics['recon_relative_error'],
                "vn_relative_error": vn_metrics['recon_relative_error'],
                
                # Sparsity (L0)
                "nl_l0_mean": nl_metrics['l0_mean'],
                "vn_l0_mean": vn_metrics['l0_mean'],
                "nl_l0_fraction": nl_metrics['l0_fraction'],
                "vn_l0_fraction": vn_metrics['l0_fraction'],
                
                # Sparsity (L1)
                "nl_l1_mean": nl_metrics['l1_mean'],
                "vn_l1_mean": vn_metrics['l1_mean'],
                
                # Top-K Concentration
                "nl_topk_10": nl_metrics['topk_10_concentration'],
                "vn_topk_10": vn_metrics['topk_10_concentration'],
                "nl_topk_50": nl_metrics['topk_50_concentration'],
                "vn_topk_50": vn_metrics['topk_50_concentration'],
                "nl_topk_100": nl_metrics['topk_100_concentration'],
                "vn_topk_100": vn_metrics['topk_100_concentration'],
                "nl_topk_500": nl_metrics['topk_500_concentration'],
                "vn_topk_500": vn_metrics['topk_500_concentration'],
                
                # Token-normalized
                "nl_features_per_token": nl_metrics['features_per_token'],
                "vn_features_per_token": vn_metrics['features_per_token'],
                "nl_intensity_per_token": nl_metrics['intensity_per_token'],
                "vn_intensity_per_token": vn_metrics['intensity_per_token'],
                
                # Feature counts
                "nl_active_features": nl_metrics['n_active_features'],
                "vn_active_features": vn_metrics['n_active_features'],
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n✓ Processed {len(results)}/{total} test cases successfully")
    
    # Analyze results
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    
    if len(results) == 0:
        print("  ⚠ No results to analyze")
        return
    
    # Overall statistics
    avg_purity_improvement = sum(r['purity_improvement_pct'] for r in results) / len(results)
    avg_recon_improvement = sum(r['recon_improvement_pct'] for r in results) / len(results)
    avg_nl_recon = sum(r['nl_recon_loss_mean'] for r in results) / len(results)
    avg_vn_recon = sum(r['vn_recon_loss_mean'] for r in results) / len(results)
    avg_token_ratio = sum(r['token_ratio'] for r in results) / len(results)
    
    print(f"\nOverall Statistics (n={len(results)}):")
    print(f"  Average Purity Improvement:    {avg_purity_improvement:+.1f}%")
    print(f"  Average Recon Improvement:     {avg_recon_improvement:+.1f}%")
    print(f"  Average NL Recon Loss:         {avg_nl_recon:.2f}")
    print(f"  Average VN Recon Loss:         {avg_vn_recon:.2f}")
    print(f"  Average Token Ratio (VN/NL):   {avg_token_ratio:.2f}")
    
    # By complexity
    print(f"\nBy Complexity:")
    for complexity in ['simple', 'medium', 'complex']:
        comp_results = [r for r in results if r['complexity'] == complexity]
        if len(comp_results) == 0:
            continue
        
        comp_purity_imp = sum(r['purity_improvement_pct'] for r in comp_results) / len(comp_results)
        comp_recon_imp = sum(r['recon_improvement_pct'] for r in comp_results) / len(comp_results)
        
        print(f"  {complexity:10} (n={len(comp_results):2}): Purity {comp_purity_imp:+.1f}%, Recon {comp_recon_imp:+.1f}%")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Convert to Parquet format
    metrics_data = {key: [r[key] for r in results] for key in results[0].keys()}
    metrics_file = storage.write_metrics(metrics_data)
    print(f"  ✓ Saved {len(results)} records to {metrics_file.name}")
    
    # By complexity aggregates
    by_complexity = {}
    for complexity in ['simple', 'medium', 'complex']:
        comp_results = [r for r in results if r['complexity'] == complexity]
        if len(comp_results) > 0:
            by_complexity[complexity] = {
                "count": len(comp_results),
                "avg_purity_improvement_pct": float(sum(r['purity_improvement_pct'] for r in comp_results) / len(comp_results)),
                "avg_recon_improvement_pct": float(sum(r['recon_improvement_pct'] for r in comp_results) / len(comp_results)),
                "avg_nl_recon_loss": float(sum(r['nl_recon_loss_mean'] for r in comp_results) / len(comp_results)),
                "avg_vn_recon_loss": float(sum(r['vn_recon_loss_mean'] for r in comp_results) / len(comp_results)),
            }
    
    # By category aggregates
    by_category = {}
    categories = set(r['category'] for r in results)
    for category in categories:
        cat_results = [r for r in results if r['category'] == category]
        if len(cat_results) > 0:
            by_category[category] = {
                "count": len(cat_results),
                "avg_purity_improvement_pct": float(sum(r['purity_improvement_pct'] for r in cat_results) / len(cat_results)),
                "avg_recon_improvement_pct": float(sum(r['recon_improvement_pct'] for r in cat_results) / len(cat_results)),
            }
    
    # Save metadata
    experiment_end_time = datetime.now().isoformat()
    metadata = {
        "experiment_type": "vn_extended_metrics",
        "specification_version": "0.3.0",
        "timing": {
            "start": experiment_start_time,
            "end": experiment_end_time,
        },
        "configuration": CONFIG,
        "device": str(device),
        "model": "gemma-2-2b",
        "sae_release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_5/width_16k/canonical",
        "summary_statistics": {
            "total_cases": len(results),
            "avg_purity_improvement_pct": float(avg_purity_improvement),
            "avg_recon_improvement_pct": float(avg_recon_improvement),
            "avg_nl_recon_loss": float(avg_nl_recon),
            "avg_vn_recon_loss": float(avg_vn_recon),
            "avg_token_ratio": float(avg_token_ratio),
        },
        "by_complexity": by_complexity,
        "by_category": by_category,
    }
    manifest_file = storage.write_manifest(metadata)
    print(f"  ✓ Saved metadata to {manifest_file.name}")
    
    # Summary
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    print(f"\n1. RECONSTRUCTION LOSS (L_rec = ||x - x_hat||²):")
    print(f"   • NL Mean: {avg_nl_recon:.2f}")
    print(f"   • VN Mean: {avg_vn_recon:.2f}")
    print(f"   • Improvement: {avg_recon_improvement:+.1f}%")
    
    print(f"\n2. TOKEN EFFICIENCY:")
    print(f"   • VN uses {avg_token_ratio:.2f}x tokens compared to NL")
    
    print(f"\n3. PURITY (entropy+SNR based):")
    print(f"   • Average Improvement: {avg_purity_improvement:+.1f}%")
    
    # Cleanup
    print("\n" + "="*80)
    print("CLEANUP")
    print("="*80)
    engine.cleanup()
    print("✓ Cleanup complete")
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)


if __name__ == "__main__":
    run_experiment()

