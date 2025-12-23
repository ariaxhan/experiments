"""●COMPONENT|Ψ:vn_comprehensive_experiments_protocol|Ω:validate_vn_syntax_effectiveness_via_sae

Comprehensive Vector-Native Experiments Runner

Runs large-scale experiments using the test cases library to validate
Vector-Native syntax effectiveness across diverse domains and complexities.

Follows Vector-Native Language Specification v0.2.0.

Methodology:
1. Load Gemma-2-2b model and GemmaScope SAE (layer 5, 16k features)
2. For each NL/VN test case pair:
   - Extract feature activations from both texts
   - Calculate spectral purity metrics
   - Identify unique features
   - Decode feature meanings (if configured)
3. Store results in Specimen Vault format:
   - Metrics: Per-case and aggregate statistics (Parquet)
   - Metadata: Experiment configuration and summary (msgpack)

Dependencies: transformer_lens, sae_lens, torch (all self-contained in protocols/)
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import torch

from protocols.storage import SpecimenStorage
from tests.vn_test_cases import (
    TEST_CASES,
    get_test_cases_by_category,
    get_test_case_stats,
    get_all_categories
)
from engines.universal_spectroscopy import UniversalSpectroscopyEngine, get_device


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

CONFIG = {
    "run_all_test_cases": True,  # Run ALL test cases (not sampling)
    "decode_top_features": 10,  # Number of features to decode
    "save_detailed_results": True,  # Save full feature analysis
    "categories_to_test": None,  # None = all categories, or specify list
}


# ============================================================================
# SECTION 2: HELPER FUNCTIONS
# ============================================================================

def decode_features(
    engine: UniversalSpectroscopyEngine,
    feature_indices: List[int],
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """●METHOD|input:engine_list_int|output:list_dict|operation:decode_feature_meanings_via_vocab
    
    Decode feature meanings by projecting through W_dec and unembedding.
    
    Args:
        engine: UniversalSpectroscopyEngine instance
        feature_indices: List of feature IDs to decode
        top_k: Number of top tokens to return per feature
        
    Returns:
        List of dicts with feature_id and decoded words
    """
    if not hasattr(engine.sae_adapter.sae, 'W_dec'):
        return []
    
    translations = []
    for feat_id in feature_indices[:top_k]:
        try:
            # Get feature direction from SAE decoder
            feature_direction = engine.sae_adapter.sae.W_dec[feat_id]
            
            # Ensure dtype consistency with model (fix for float16/float32 mismatch)
            if hasattr(engine.model, 'W_U'):
                target_dtype = engine.model.W_U.dtype
            else:
                target_dtype = next(engine.model.parameters()).dtype
            
            feature_direction = feature_direction.to(dtype=target_dtype, device=engine.device)
            
            # Project through unembedding to get logits
            logits = engine.model.unembed(feature_direction)
            top_token_ids = logits.argsort(descending=True)[:3]
            top_words = engine.model.to_str_tokens(top_token_ids)
            
            translations.append({
                "feature_id": int(feat_id),
                "words": top_words[:3]
            })
        except Exception as e:
            print(f"  Warning: Could not decode feature {feat_id}: {e}")
            continue
    
    return translations


def get_unique_features(
    spec1: Any,  # Spectrum object
    spec2: Any,  # Spectrum object
    top_k: int = 10
) -> List[int]:
    """●METHOD|input:spectrum_spectrum_int|output:list_int|operation:find_features_unique_to_spec2
    
    Find features unique to spec2 (not present in spec1).
    
    Args:
        spec1: First spectrum (baseline)
        spec2: Second spectrum (to find unique features in)
        top_k: Number of top features to return
        
    Returns:
        List of feature indices unique to spec2, sorted by intensity
    """
    set1 = set(spec1.wavelengths.tolist())
    set2 = set(spec2.wavelengths.tolist())
    
    unique = set2 - set1
    if len(unique) == 0:
        return []
    
    unique_mask = torch.tensor([w in unique for w in spec2.wavelengths.tolist()])
    unique_wavelengths = spec2.wavelengths[unique_mask]
    unique_intensities = spec2.intensities[unique_mask]
    
    sorted_indices = torch.argsort(unique_intensities, descending=True)
    top_features = unique_wavelengths[sorted_indices[:top_k]]
    
    return top_features.tolist()


def calculate_metrics(
    engine: UniversalSpectroscopyEngine,
    nl_spectrum: Any,
    vn_spectrum: Any
) -> Dict[str, float]:
    """●METHOD|input:engine_spectrum_spectrum|output:dict|operation:calculate_purity_and_feature_metrics
    
    Calculate comprehensive metrics comparing NL and VN spectra.
    
    Args:
        engine: UniversalSpectroscopyEngine instance
        nl_spectrum: Natural language spectrum
        vn_spectrum: Vector-Native spectrum
        
    Returns:
        Dict with purity, feature counts, overlap metrics
    """
    nl_purity = engine.calculate_purity(nl_spectrum)
    vn_purity = engine.calculate_purity(vn_spectrum)
    
    improvement = (vn_purity - nl_purity) / nl_purity * 100 if nl_purity > 0 else 0
    
    nl_features = set(nl_spectrum.wavelengths.tolist())
    vn_features = set(vn_spectrum.wavelengths.tolist())
    
    overlap = len(nl_features & vn_features)
    total_unique = len(nl_features | vn_features)
    overlap_pct = (overlap / total_unique * 100) if total_unique > 0 else 0
    
    feature_reduction = (len(nl_features) - len(vn_features)) / len(nl_features) * 100 if len(nl_features) > 0 else 0
    
    return {
        "nl_purity": float(nl_purity),
        "vn_purity": float(vn_purity),
        "purity_improvement_pct": float(improvement),
        "nl_features": len(nl_features),
        "vn_features": len(vn_features),
        "feature_reduction_pct": float(feature_reduction),
        "overlap_count": overlap,
        "overlap_pct": float(overlap_pct)
    }


# ============================================================================
# SECTION 3: MAIN EXPERIMENT
# ============================================================================

def run_experiment() -> None:
    """●METHOD|input:None|output:None|operation:execute_vn_comprehensive_experiments
    
    Main entry point for comprehensive Vector-Native experiments.
    
    Workflow:
    1. Initialize storage and instruments
    2. Select test cases
    3. Process NL/VN pairs and calculate metrics
    4. Save results to Specimen Vault (Parquet + msgpack)
    """
    experiment_start_time = datetime.now().isoformat()
    
    # Initialize storage (creates new run automatically)
    specimen_path = Path(__file__).parent
    storage = SpecimenStorage(specimen_path)
    print(f"●PROCESS|operation:vn_comprehensive_experiments|phase:starting")
    print(f"  Specimen: {specimen_path.name}")
    print(f"  Run ID: {storage.run_id}")
    print(f"  Run directory: {storage.run_path}")
    
    # Show previous runs if any
    previous_runs = storage.list_runs()
    if len(previous_runs) > 1:  # More than current run
        print(f"  Previous runs: {len(previous_runs) - 1}")
        print(f"    Latest: {previous_runs[1] if len(previous_runs) > 1 else 'N/A'}")
    
    print("\n" + "="*80)
    print("COMPREHENSIVE VECTOR-NATIVE EXPERIMENTS")
    print("Using Test Cases Library with 100+ Diverse Examples")
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
            print(f"\nRunning ALL {len(selected_cases)} test cases from ALL categories")
        else:
            selected_cases = [(k, v) for k, v in TEST_CASES.items() 
                            if v['category'] in CONFIG["categories_to_test"]]
            print(f"\nRunning ALL {len(selected_cases)} test cases from selected categories")
    else:
        # Sampling logic (for backwards compatibility)
        import random
        selected_cases = []
        categories = CONFIG["categories_to_test"] if CONFIG["categories_to_test"] else get_all_categories()
        
        for category in categories:
            category_cases = get_test_cases_by_category(category)
            sample_size = CONFIG.get("sample_size_per_category", 5)
            selected = random.sample(list(category_cases.keys()), 
                                    min(sample_size, len(category_cases)))
            selected_cases.extend([(k, TEST_CASES[k]) for k in selected])
        
        print(f"\nSelected {len(selected_cases)} test cases (sampling)")
    
    # Print breakdown
    categories_in_selection = set(case['category'] for _, case in selected_cases)
    print(f"\nBreakdown by category:")
    for category in sorted(categories_in_selection):
        count = sum(1 for _, case in selected_cases if case['category'] == category)
        print(f"  {category}: {count} cases")
    
    print(f"\nBreakdown by complexity:")
    for complexity in ['simple', 'medium', 'complex']:
        count = sum(1 for _, case in selected_cases if case['complexity'] == complexity)
        if count > 0:
            print(f"  {complexity}: {count} cases")
    
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
        print(f"  Description: {case_data['description']}")
        
        try:
            # Process both versions
            nl_spectrum = engine.process(case_data['nl'])
            vn_spectrum = engine.process(case_data['vn'])
            
            # Calculate metrics
            metrics = calculate_metrics(engine, nl_spectrum, vn_spectrum)
            
            print(f"  NL Purity: {metrics['nl_purity']:.4f} ({metrics['nl_features']} features)")
            print(f"  VN Purity: {metrics['vn_purity']:.4f} ({metrics['vn_features']} features)")
            print(f"  Improvement: {metrics['purity_improvement_pct']:+.1f}%")
            print(f"  Feature Reduction: {metrics['feature_reduction_pct']:.1f}%")
            
            # Get unique features
            vn_unique = get_unique_features(nl_spectrum, vn_spectrum, top_k=CONFIG["decode_top_features"])
            nl_unique = get_unique_features(vn_spectrum, nl_spectrum, top_k=CONFIG["decode_top_features"])
            
            # Decode features if configured
            decoded_vn = []
            if CONFIG["save_detailed_results"] and len(vn_unique) > 0:
                decoded_vn = decode_features(engine, vn_unique, top_k=5)
                if len(decoded_vn) > 0:
                    feature_ids = ', '.join(f"#{d['feature_id']}" for d in decoded_vn[:3])
                    print(f"  VN Unique Features (sample): {feature_ids}")
            
            # Store results (will be converted to Parquet format)
            result = {
                "case_id": case_id,
                "category": case_data['category'],
                "complexity": case_data['complexity'],
                "description": case_data['description'],
                "nl_text": case_data['nl'],
                "vn_text": case_data['vn'],
                "expected_reduction": case_data.get('expected_reduction', 0.0),
                "nl_purity": metrics['nl_purity'],
                "vn_purity": metrics['vn_purity'],
                "purity_improvement_pct": metrics['purity_improvement_pct'],
                "nl_features": metrics['nl_features'],
                "vn_features": metrics['vn_features'],
                "feature_reduction_pct": metrics['feature_reduction_pct'],
                "overlap_count": metrics['overlap_count'],
                "overlap_pct": metrics['overlap_pct'],
                "vn_unique_count": len(vn_unique),
                "nl_unique_count": len(nl_unique),
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ Error processing case: {e}")
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
    avg_feature_reduction = sum(r['feature_reduction_pct'] for r in results) / len(results)
    avg_nl_purity = sum(r['nl_purity'] for r in results) / len(results)
    avg_vn_purity = sum(r['vn_purity'] for r in results) / len(results)
    
    print(f"\nOverall Statistics:")
    print(f"  Average NL Purity:          {avg_nl_purity:.4f}")
    print(f"  Average VN Purity:          {avg_vn_purity:.4f}")
    print(f"  Average Purity Improvement: {avg_purity_improvement:+.1f}%")
    print(f"  Average Feature Reduction:  {avg_feature_reduction:.1f}%")
    
    # By category
    print(f"\nBy Category:")
    categories_in_results = set(r['category'] for r in results)
    for category in sorted(categories_in_results):
        cat_results = [r for r in results if r['category'] == category]
        if len(cat_results) == 0:
            continue
        
        cat_avg_improvement = sum(r['purity_improvement_pct'] for r in cat_results) / len(cat_results)
        cat_avg_reduction = sum(r['feature_reduction_pct'] for r in cat_results) / len(cat_results)
        
        print(f"  {category:20} | Improvement: {cat_avg_improvement:+.1f}% | Reduction: {cat_avg_reduction:.1f}%")
    
    # By complexity
    print(f"\nBy Complexity:")
    for complexity in ['simple', 'medium', 'complex']:
        comp_results = [r for r in results if r['complexity'] == complexity]
        if len(comp_results) == 0:
            continue
        
        comp_avg_improvement = sum(r['purity_improvement_pct'] for r in comp_results) / len(comp_results)
        comp_avg_reduction = sum(r['feature_reduction_pct'] for r in comp_results) / len(comp_results)
        
        print(f"  {complexity:10} | Improvement: {comp_avg_improvement:+.1f}% | Reduction: {comp_avg_reduction:.1f}%")
    
    # Top performers
    print(f"\nTop 5 Improvements:")
    top_improvements = sorted(results, key=lambda x: x['purity_improvement_pct'], reverse=True)[:5]
    for i, r in enumerate(top_improvements, 1):
        print(f"  {i}. {r['case_id']:30} | {r['purity_improvement_pct']:+.1f}% | {r['category']}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Convert results to Parquet format (list of dicts with same keys)
    print("  Writing metrics to Parquet...")
    metrics_data = {
        "case_id": [r["case_id"] for r in results],
        "category": [r["category"] for r in results],
        "complexity": [r["complexity"] for r in results],
        "description": [r["description"] for r in results],
        "nl_text": [r["nl_text"] for r in results],
        "vn_text": [r["vn_text"] for r in results],
        "expected_reduction": [r["expected_reduction"] for r in results],
        "nl_purity": [r["nl_purity"] for r in results],
        "vn_purity": [r["vn_purity"] for r in results],
        "purity_improvement_pct": [r["purity_improvement_pct"] for r in results],
        "nl_features": [r["nl_features"] for r in results],
        "vn_features": [r["vn_features"] for r in results],
        "feature_reduction_pct": [r["feature_reduction_pct"] for r in results],
        "overlap_count": [r["overlap_count"] for r in results],
        "overlap_pct": [r["overlap_pct"] for r in results],
        "vn_unique_count": [r["vn_unique_count"] for r in results],
        "nl_unique_count": [r["nl_unique_count"] for r in results],
    }
    metrics_file = storage.write_metrics(metrics_data)
    print(f"    ✓ Saved {len(results)} test case records")
    print(f"      Run: {storage.run_id}")
    print(f"      File: {metrics_file.name}")
    
    # Calculate aggregate statistics for metadata
    by_category = {}
    for category in categories_in_results:
        cat_results = [r for r in results if r['category'] == category]
        by_category[category] = {
            "count": len(cat_results),
            "avg_improvement_pct": float(sum(r['purity_improvement_pct'] for r in cat_results) / len(cat_results)) if len(cat_results) > 0 else 0.0,
            "avg_reduction_pct": float(sum(r['feature_reduction_pct'] for r in cat_results) / len(cat_results)) if len(cat_results) > 0 else 0.0,
        }
    
    by_complexity = {}
    for complexity in ['simple', 'medium', 'complex']:
        comp_results = [r for r in results if r['complexity'] == complexity]
        if len(comp_results) > 0:
            by_complexity[complexity] = {
                "count": len(comp_results),
                "avg_improvement_pct": float(sum(r['purity_improvement_pct'] for r in comp_results) / len(comp_results)),
                "avg_reduction_pct": float(sum(r['feature_reduction_pct'] for r in comp_results) / len(comp_results)),
            }
    
    # Save experiment metadata
    experiment_end_time = datetime.now().isoformat()
    metadata = {
        "experiment_type": "vn_comprehensive_experiments",
        "specification_version": "0.2.0",
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
            "avg_nl_purity": float(avg_nl_purity),
            "avg_vn_purity": float(avg_vn_purity),
            "avg_purity_improvement_pct": float(avg_purity_improvement),
            "avg_feature_reduction_pct": float(avg_feature_reduction),
        },
        "by_category": by_category,
        "by_complexity": by_complexity,
    }
    manifest_file = storage.write_manifest(metadata)
    print(f"    ✓ Saved experiment metadata")
    print(f"      Run: {storage.run_id}")
    print(f"      File: {manifest_file.name}")
    
    # Summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    print(f"\n{'METRIC':<40} | {'VALUE':<20}")
    print("-" * 80)
    print(f"{'Test Cases Processed':<40} | {len(results):<20}")
    print(f"{'Average Purity Improvement':<40} | {avg_purity_improvement:+.1f}%")
    print(f"{'Average Feature Reduction':<40} | {avg_feature_reduction:.1f}%")
    print(f"{'Average VN Purity':<40} | {avg_vn_purity:.4f}")
    print(f"{'Average NL Purity':<40} | {avg_nl_purity:.4f}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    print(f"\n1. OVERALL PERFORMANCE:")
    print(f"   • VN syntax shows {avg_purity_improvement:+.1f}% average purity improvement")
    print(f"   • {avg_feature_reduction:.1f}% average reduction in feature activations")
    print(f"   • Tested across {len(results)} diverse cases")
    
    print(f"\n2. BY CATEGORY:")
    for category in sorted(by_category.keys()):
        cat_data = by_category[category]
        if cat_data['count'] > 0:
            print(f"   • {category}: {cat_data['avg_improvement_pct']:+.1f}% improvement, {cat_data['avg_reduction_pct']:.1f}% reduction")
    
    print(f"\n3. BY COMPLEXITY:")
    for complexity in ['simple', 'medium', 'complex']:
        if complexity in by_complexity:
            comp_data = by_complexity[complexity]
            print(f"   • {complexity}: {comp_data['avg_improvement_pct']:+.1f}% improvement, {comp_data['avg_reduction_pct']:.1f}% reduction")
    
    print(f"\n4. TOP PERFORMERS:")
    for i, r in enumerate(top_improvements[:3], 1):
        print(f"   • {r['description']}: {r['purity_improvement_pct']:+.1f}% improvement")
    
    # Cleanup
    print("\n" + "="*80)
    print("CLEANUP")
    print("="*80)
    print(">> Cleaning up...")
    engine.cleanup()
    print("✓ Cleanup complete")
    
    print("\n" + "="*80)
    print("EXPERIMENTS COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("  1. Run: python scripts/index_vault.py")
    print("  2. Query results via VaultQuery")
    print("  3. Inspect metrics: storage.read_metrics()")
    print("  4. Analyze feature patterns across categories")
    print("  5. Explore decoded features on Neuronpedia")
    print()


if __name__ == "__main__":
    run_experiment()
