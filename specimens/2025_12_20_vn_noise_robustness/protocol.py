"""●COMPONENT|Ψ:noise_robustness_protocol|Ω:test_vn_fault_tolerance_via_noise_injection

Experiment F: Noise Robustness Test

HYPOTHESIS: VN's explicit structure creates fault-tolerant representations.
NL relies on fragile grammatical/contextual cues that break under noise.
VN's structural anchors (●, |, :) maintain semantic integrity.

METHODOLOGY:
1. Inject noise into NL and VN texts at various rates (0%, 5%, 10%, 15%, 20%, 25%)
2. For VN, preserve structural tokens (●, |, :) - they are the "anchors"
3. Process noisy variants through SAE at layer 5
4. Calculate metrics:
   - Semantic retention: Feature overlap with clean version
   - Reconstruction stability: How much does loss increase?
   - Purity degradation: How much does purity drop?
   - Critical feature survival: % of top-20 features that survive
5. Compare NL vs VN robustness across noise levels
"""

import random
import string
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch

from protocols.storage import SpecimenStorage
from tests.vn_test_cases import (
    TEST_CASES,
    get_test_cases_by_complexity,
    get_test_case_stats
)
from engines.universal_spectroscopy import UniversalSpectroscopyEngine, get_device


# ============================================================================
# SECTION 1: CONFIGURATION
# ============================================================================

CONFIG = {
    "test_cases_per_complexity": 10,  # 10 simple, 10 medium, 10 complex = 30 total
    "noise_rates": [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],  # 0% to 25%
    "variants_per_noise_rate": 5,  # Generate 5 noisy variants per (case, noise_rate)
    "top_k_critical_features": 20,  # Top features to track survival
    "noise_types": ["character_swap", "character_drop", "character_insert", "mixed"],
    "preserve_vn_structure": True,  # Preserve ●, |, : in VN
}


# ============================================================================
# SECTION 2: NOISE INJECTION FUNCTIONS
# ============================================================================

# Adjacent keyboard characters for realistic swaps
KEYBOARD_ADJACENT = {
    'q': ['w', 'a'], 'w': ['q', 'e', 's'], 'e': ['w', 'r', 'd'], 'r': ['e', 't', 'f'],
    't': ['r', 'y', 'g'], 'y': ['t', 'u', 'h'], 'u': ['y', 'i', 'j'], 'i': ['u', 'o', 'k'],
    'o': ['i', 'p', 'l'], 'p': ['o', '['],
    'a': ['q', 's', 'z'], 's': ['a', 'd', 'w', 'x'], 'd': ['s', 'f', 'e', 'c'],
    'f': ['d', 'g', 'r', 'v'], 'g': ['f', 'h', 't', 'b'], 'h': ['g', 'j', 'y', 'n'],
    'j': ['h', 'k', 'u', 'm'], 'k': ['j', 'l', 'i'], 'l': ['k', ';', 'o'],
    'z': ['a', 'x'], 'x': ['z', 'c', 's'], 'c': ['x', 'v', 'd'], 'v': ['c', 'b', 'f'],
    'b': ['v', 'n', 'g'], 'n': ['b', 'm', 'h'], 'm': ['n', 'j'],
}


def get_adjacent_char(char: str) -> str:
    """●METHOD|input:str|output:str|operation:get_random_adjacent_keyboard_char"""
    char_lower = char.lower()
    if char_lower in KEYBOARD_ADJACENT:
        candidates = KEYBOARD_ADJACENT[char_lower]
        adjacent = random.choice(candidates)
        return adjacent.upper() if char.isupper() else adjacent
    return char


def inject_noise(
    text: str,
    noise_rate: float,
    noise_type: str,
    preserve_vn_structure: bool = False,
    seed: Optional[int] = None
) -> str:
    """●METHOD|input:str_float_str_bool_int|output:str|operation:inject_noise_into_text
    
    Inject noise into text at specified rate.
    
    Args:
        text: Input text
        noise_rate: Fraction of characters to modify (0.0 to 1.0)
        noise_type: Type of noise ("character_swap", "character_drop", "character_insert", "mixed")
        preserve_vn_structure: If True, preserve VN structural tokens (●, |, :)
        seed: Random seed for reproducibility
        
    Returns:
        Noisy text
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    if noise_rate == 0.0:
        return text
    
    # Identify structural tokens to preserve (for VN)
    structural_chars = {'●', '|', ':'} if preserve_vn_structure else set()
    
    # Convert to list for mutation
    chars = list(text)
    n_chars = len(chars)
    
    if n_chars == 0:
        return text
    
    # Calculate number of positions to modify
    n_modify = max(1, int(n_chars * noise_rate))
    
    # Get positions that can be modified (exclude structural chars)
    modifiable_positions = [
        i for i in range(n_chars)
        if chars[i] not in structural_chars
    ]
    
    if len(modifiable_positions) == 0:
        return text  # All chars are structural, can't modify
    
    # Sample positions to modify
    positions_to_modify = random.sample(modifiable_positions, min(n_modify, len(modifiable_positions)))
    
    # Apply noise based on type
    if noise_type == "character_swap":
        for pos in positions_to_modify:
            chars[pos] = get_adjacent_char(chars[pos])
    
    elif noise_type == "character_drop":
        # Sort positions in reverse to avoid index shifting
        for pos in sorted(positions_to_modify, reverse=True):
            if 0 <= pos < len(chars):
                chars.pop(pos)
    
    elif noise_type == "character_insert":
        # Sort positions in forward order - insertions shift later indices, so we need to process from start
        # But we need to track shifts to avoid invalid positions
        sorted_positions = sorted(positions_to_modify)
        shift = 0
        for pos in sorted_positions:
            adjusted_pos = pos + shift
            if 0 <= adjusted_pos <= len(chars):
                random_char = random.choice(string.ascii_letters + string.digits + " ")
                chars.insert(adjusted_pos, random_char)
                shift += 1  # Each insertion shifts subsequent positions forward
    
    elif noise_type == "mixed":
        # For mixed operations, we need to process in a way that accounts for shifting
        # Strategy: process swaps first (no shifting), then drops (reverse order), then inserts (forward with shift tracking)
        swap_positions = []
        drop_positions = []
        insert_positions = []
        
        # Assign random noise type to each position
        for pos in positions_to_modify:
            noise_choice = random.choice(["swap", "drop", "insert"])
            if noise_choice == "swap":
                swap_positions.append(pos)
            elif noise_choice == "drop":
                drop_positions.append(pos)
            elif noise_choice == "insert":
                insert_positions.append(pos)
        
        # Process swaps first (no index shifting)
        for pos in swap_positions:
            if 0 <= pos < len(chars):
                chars[pos] = get_adjacent_char(chars[pos])
        
        # Process drops in reverse order (removes from end first)
        for pos in sorted(drop_positions, reverse=True):
            if 0 <= pos < len(chars):
                chars.pop(pos)
                # Update insert positions that come after this drop
                insert_positions = [ip if ip < pos else ip - 1 for ip in insert_positions]
        
        # Process inserts in forward order with shift tracking
        sorted_inserts = sorted(insert_positions)
        shift = 0
        for pos in sorted_inserts:
            adjusted_pos = pos + shift
            if 0 <= adjusted_pos <= len(chars):
                random_char = random.choice(string.ascii_letters + string.digits + " ")
                chars.insert(adjusted_pos, random_char)
                shift += 1
    
    return ''.join(chars)


# ============================================================================
# SECTION 3: METRIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_semantic_retention(
    clean_features: set,
    noisy_features: set
) -> float:
    """●METHOD|input:set_set|output:float|operation:calculate_feature_overlap_ratio
    
    Calculate what fraction of clean features survive in noisy version.
    
    Returns:
        Retention ratio (0.0 to 1.0)
    """
    if len(clean_features) == 0:
        return 0.0
    overlap = len(clean_features & noisy_features)
    return overlap / len(clean_features)


def calculate_reconstruction_stability(
    clean_loss: float,
    noisy_loss: float
) -> float:
    """●METHOD|input:float_float|output:float|operation:calculate_loss_ratio
    
    How much does reconstruction loss increase with noise?
    
    Returns:
        Stability ratio (lower = more stable, 1.0 = no increase)
    """
    if clean_loss == 0:
        return 1.0 if noisy_loss == 0 else 0.0
    return clean_loss / noisy_loss  # Inverted: higher = more stable


def calculate_purity_retention(
    clean_purity: float,
    noisy_purity: float
) -> float:
    """●METHOD|input:float_float|output:float|operation:calculate_purity_ratio
    
    How much does purity degrade with noise?
    
    Returns:
        Purity retention ratio (1.0 = no degradation)
    """
    if clean_purity == 0:
        return 0.0
    return noisy_purity / clean_purity


def calculate_critical_feature_survival(
    clean_top_features: List[int],
    noisy_features: set
) -> float:
    """●METHOD|input:list_set|output:float|operation:calculate_top_feature_survival_rate
    
    What % of top-K features from clean version survive in noisy?
    
    Returns:
        Survival rate (0.0 to 1.0)
    """
    if len(clean_top_features) == 0:
        return 0.0
    survived = sum(1 for feat in clean_top_features if feat in noisy_features)
    return survived / len(clean_top_features)


def get_top_features(spectrum: Any, top_k: int) -> List[int]:
    """●METHOD|input:spectrum_int|output:list_int|operation:extract_top_k_features_by_intensity"""
    if len(spectrum) == 0:
        return []
    
    # Sort by intensity (descending)
    intensities = spectrum.intensities
    wavelengths = spectrum.wavelengths
    
    sorted_indices = torch.argsort(intensities, descending=True)
    top_wavelengths = wavelengths[sorted_indices[:top_k]]
    
    return top_wavelengths.tolist()


def calculate_reconstruction_loss(
    engine: UniversalSpectroscopyEngine,
    text: str
) -> float:
    """●METHOD|input:engine_str|output:float|operation:calculate_sae_reconstruction_mse
    
    Calculate SAE reconstruction loss (MSE between original and reconstructed activations).
    
    Returns:
        MSE loss value
    """
    if not hasattr(engine.sae_adapter.sae, 'W_dec'):
        return 0.0
    
    try:
        layer = engine.sae_adapter.layer
        if layer is None:
            return 0.0
        
        # Get original activations
        activations = engine.excitation_controller.process(
            input_text=text,
            model=engine.model,
            layer=layer,
            token_positions=None
        )
        
        # Flatten
        original = activations.view(-1, activations.shape[-1])
        
        # Encode and decode through SAE
        with torch.no_grad():
            encoded = engine.sae_adapter.sae.encode(original)
            reconstructed = engine.sae_adapter.sae.decode(encoded)
            
            # Calculate MSE loss
            mse = torch.nn.functional.mse_loss(reconstructed, original)
            
            return mse.item()
            
    except Exception as e:
        print(f"  Warning: Could not calculate reconstruction loss: {e}")
        return 0.0


# ============================================================================
# SECTION 4: MAIN EXPERIMENT
# ============================================================================

def run_experiment() -> None:
    """●METHOD|input:None|output:None|operation:execute_noise_robustness_experiment
    
    Main entry point for noise robustness experiment.
    """
    experiment_start_time = datetime.now().isoformat()
    
    # Initialize storage
    specimen_path = Path(__file__).parent
    storage = SpecimenStorage(specimen_path)
    print(f"●PROCESS|operation:noise_robustness_experiment|phase:starting")
    print(f"  Storage initialized at: {specimen_path}")
    
    print("\n" + "="*80)
    print("EXPERIMENT F: NOISE ROBUSTNESS TEST")
    print("Testing VN fault tolerance vs NL fragility under noise")
    print("="*80)
    
    # Initialize device and engine
    device = get_device()
    print(f"\nUsing device: {device}")
    
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
    
    selected_cases = []
    for complexity in ["simple", "medium", "complex"]:
        complexity_cases = get_test_cases_by_complexity(complexity)
        case_ids = list(complexity_cases.keys())
        # Sample randomly
        random.shuffle(case_ids)
        sampled = case_ids[:CONFIG["test_cases_per_complexity"]]
        selected_cases.extend([(case_id, complexity_cases[case_id]) for case_id in sampled])
    
    print(f"\nSelected {len(selected_cases)} test cases:")
    print(f"  Simple: {sum(1 for _, c in selected_cases if c['complexity'] == 'simple')}")
    print(f"  Medium: {sum(1 for _, c in selected_cases if c['complexity'] == 'medium')}")
    print(f"  Complex: {sum(1 for _, c in selected_cases if c['complexity'] == 'complex')}")
    
    # Run experiments
    print("\n" + "="*80)
    print("RUNNING NOISE ROBUSTNESS EXPERIMENTS")
    print("="*80)
    print(f"Noise rates: {CONFIG['noise_rates']}")
    print(f"Variants per noise rate: {CONFIG['variants_per_noise_rate']}")
    print(f"Total runs: {len(selected_cases)} cases × {len(CONFIG['noise_rates'])} rates × {CONFIG['variants_per_noise_rate']} variants × 2 (NL/VN)")
    
    results = []
    total_runs = len(selected_cases) * len(CONFIG['noise_rates']) * CONFIG['variants_per_noise_rate'] * 2
    processed = 0
    
    for case_id, case_data in selected_cases:
        print(f"\n[{processed // (len(CONFIG['noise_rates']) * CONFIG['variants_per_noise_rate'] * 2) + 1}/{len(selected_cases)}] Processing: {case_id}")
        print(f"  Complexity: {case_data['complexity']} | Category: {case_data['category']}")
        
        nl_clean = case_data['nl']
        vn_clean = case_data['vn']
        
        # Process clean versions once
        try:
            nl_clean_spectrum = engine.process(nl_clean)
            vn_clean_spectrum = engine.process(vn_clean)
            
            nl_clean_features = set(nl_clean_spectrum.wavelengths.tolist())
            vn_clean_features = set(vn_clean_spectrum.wavelengths.tolist())
            
            nl_clean_purity = engine.calculate_purity(nl_clean_spectrum)
            vn_clean_purity = engine.calculate_purity(vn_clean_spectrum)
            
            nl_clean_top_features = get_top_features(nl_clean_spectrum, CONFIG['top_k_critical_features'])
            vn_clean_top_features = get_top_features(vn_clean_spectrum, CONFIG['top_k_critical_features'])
            
            nl_clean_loss = calculate_reconstruction_loss(engine, nl_clean)
            vn_clean_loss = calculate_reconstruction_loss(engine, vn_clean)
            
        except Exception as e:
            print(f"  ✗ Error processing clean versions: {e}")
            continue
        
        # Test each noise rate
        for noise_rate in CONFIG['noise_rates']:
            # Generate multiple noisy variants
            for variant_idx in range(CONFIG['variants_per_noise_rate']):
                seed = hash(f"{case_id}_{noise_rate}_{variant_idx}") % (2**31)
                
                # Test NL
                try:
                    nl_noisy = inject_noise(
                        nl_clean,
                        noise_rate=noise_rate,
                        noise_type="mixed",
                        preserve_vn_structure=False,
                        seed=seed
                    )
                    nl_noisy_spectrum = engine.process(nl_noisy)
                    nl_noisy_features = set(nl_noisy_spectrum.wavelengths.tolist())
                    nl_noisy_purity = engine.calculate_purity(nl_noisy_spectrum)
                    nl_noisy_loss = calculate_reconstruction_loss(engine, nl_noisy)
                    
                    nl_metrics = {
                        "case_id": case_id,
                        "complexity": case_data['complexity'],
                        "category": case_data['category'],
                        "encoding": "nl",
                        "noise_rate": noise_rate,
                        "variant_idx": variant_idx,
                        "semantic_retention": calculate_semantic_retention(nl_clean_features, nl_noisy_features),
                        "reconstruction_stability": calculate_reconstruction_stability(nl_clean_loss, nl_noisy_loss),
                        "purity_retention": calculate_purity_retention(nl_clean_purity, nl_noisy_purity),
                        "critical_feature_survival": calculate_critical_feature_survival(nl_clean_top_features, nl_noisy_features),
                        "clean_purity": float(nl_clean_purity),
                        "noisy_purity": float(nl_noisy_purity),
                        "clean_loss": float(nl_clean_loss),
                        "noisy_loss": float(nl_noisy_loss),
                        "clean_features_count": len(nl_clean_features),
                        "noisy_features_count": len(nl_noisy_features),
                    }
                    results.append(nl_metrics)
                    processed += 1
                    
                except Exception as e:
                    print(f"  ✗ Error processing NL noisy variant: {e}")
                
                # Test VN
                try:
                    vn_noisy = inject_noise(
                        vn_clean,
                        noise_rate=noise_rate,
                        noise_type="mixed",
                        preserve_vn_structure=True,  # Preserve structural tokens
                        seed=seed
                    )
                    vn_noisy_spectrum = engine.process(vn_noisy)
                    vn_noisy_features = set(vn_noisy_spectrum.wavelengths.tolist())
                    vn_noisy_purity = engine.calculate_purity(vn_noisy_spectrum)
                    vn_noisy_loss = calculate_reconstruction_loss(engine, vn_noisy)
                    
                    vn_metrics = {
                        "case_id": case_id,
                        "complexity": case_data['complexity'],
                        "category": case_data['category'],
                        "encoding": "vn",
                        "noise_rate": noise_rate,
                        "variant_idx": variant_idx,
                        "semantic_retention": calculate_semantic_retention(vn_clean_features, vn_noisy_features),
                        "reconstruction_stability": calculate_reconstruction_stability(vn_clean_loss, vn_noisy_loss),
                        "purity_retention": calculate_purity_retention(vn_clean_purity, vn_noisy_purity),
                        "critical_feature_survival": calculate_critical_feature_survival(vn_clean_top_features, vn_noisy_features),
                        "clean_purity": float(vn_clean_purity),
                        "noisy_purity": float(vn_noisy_purity),
                        "clean_loss": float(vn_clean_loss),
                        "noisy_loss": float(vn_noisy_loss),
                        "clean_features_count": len(vn_clean_features),
                        "noisy_features_count": len(vn_noisy_features),
                    }
                    results.append(vn_metrics)
                    processed += 1
                    
                    if processed % 10 == 0:
                        print(f"  Progress: {processed}/{total_runs} runs completed")
                        
                except Exception as e:
                    print(f"  ✗ Error processing VN noisy variant: {e}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    if len(results) == 0:
        print("✗ No results to save!")
        return
    
    # Convert to columnar format for Parquet
    metrics_dict = {}
    for key in results[0].keys():
        metrics_dict[key] = [r[key] for r in results]
    
    storage.write_metrics(metrics_dict)
    print(f"✓ Saved {len(results)} result rows to metrics.parquet")
    
    # Calculate summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Group by encoding and noise rate
    for encoding in ["nl", "vn"]:
        encoding_results = [r for r in results if r['encoding'] == encoding]
        print(f"\n{encoding.upper()} Results:")
        for noise_rate in CONFIG['noise_rates']:
            rate_results = [r for r in encoding_results if r['noise_rate'] == noise_rate]
            if len(rate_results) == 0:
                continue
            
            avg_retention = np.mean([r['semantic_retention'] for r in rate_results])
            avg_purity_retention = np.mean([r['purity_retention'] for r in rate_results])
            avg_feature_survival = np.mean([r['critical_feature_survival'] for r in rate_results])
            
            print(f"  Noise {noise_rate*100:.0f}%: "
                  f"Retention={avg_retention:.3f}, "
                  f"Purity={avg_purity_retention:.3f}, "
                  f"TopFeatures={avg_feature_survival:.3f}")
    
    # Update manifest
    manifest = {
        "specimen_id": "2025_12_20_vn_noise_robustness",
        "created": experiment_start_time,
        "completed": datetime.now().isoformat(),
        "taxonomy": {
            "domain": "interpretability",
            "method": "sae_analysis"
        },
        "tags": ["vn", "noise_robustness", "gemma-2-2b", "layer_5"],
        "config": CONFIG,
        "summary": {
            "total_cases": len(selected_cases),
            "total_runs": len(results),
            "noise_rates_tested": CONFIG['noise_rates'],
            "variants_per_rate": CONFIG['variants_per_noise_rate'],
        }
    }
    
    storage.write_manifest(manifest)
    print(f"\n✓ Experiment complete!")
    print(f"  Results saved to: {specimen_path / 'strata' / 'metrics.parquet'}")


if __name__ == "__main__":
    run_experiment()
