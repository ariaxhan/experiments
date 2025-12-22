"""●COMPONENT|Ψ:convergence_velocity_protocol|Ω:test_vn_convergence_speed_vs_nl

Experiment H: Convergence Velocity Test

HYPOTHESIS: VN reaches stable task representations in earlier layers.
The model "knows what to do" faster with VN because there's less 
linguistic ambiguity to resolve.

METHODOLOGY:
1. Extract residual stream activations at multiple layers [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25]
2. For each NL/VN pair, compute convergence metrics:
   - Final-layer similarity: cosine_sim(activation[L], activation[final])
   - Convergence velocity: first layer where similarity > 0.9
   - Representation stability: cosine_sim(activation[L], activation[L+1])
   - Cross-encoding convergence: cosine_sim(NL[L], VN[L])
   - Task-concept alignment: cosine_sim(activation[L], task_vector)
3. Visualize convergence curves, velocity histograms, and trajectory plots
4. Compare NL vs VN convergence patterns
"""

import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

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
    "layers_to_sample": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25],  # Gemma-2-2b has 26 layers (0-25)
    "final_layer": 25,  # Last layer for convergence comparison
    "convergence_threshold": 0.9,  # Threshold for "converged" similarity
    "stability_threshold": 0.95,  # Threshold for "stable" representation
    "stability_window": 3,  # Consecutive layers needed for stability
    "token_position": -1,  # Use last token position (where prediction happens)
}


# ============================================================================
# SECTION 2: ACTIVATION EXTRACTION
# ============================================================================

def extract_multi_layer_activations(
    engine: UniversalSpectroscopyEngine,
    text: str,
    layers: List[int],
    token_position: Optional[int] = None
) -> Dict[int, torch.Tensor]:
    """●METHOD|input:engine_str_list_int|output:dict|operation:extract_activations_from_multiple_layers
    
    Extract residual stream activations from multiple layers in a single forward pass.
    
    Args:
        engine: UniversalSpectroscopyEngine instance
        text: Input text to process
        layers: List of layer indices to extract
        token_position: Specific token position to extract (None = all, -1 = last)
        
    Returns:
        Dictionary mapping layer -> activation tensor (flattened to [hidden_dim])
    """
    model = engine.model
    device = engine.excitation_controller.device
    
    # Tokenize input
    tokens = model.to_tokens(text)
    tokens = tokens.to(device)
    
    # Run model once and cache all layers
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, return_type=None)
    
    activations_dict = {}
    for layer in layers:
        activation_key = f"blocks.{layer}.hook_resid_post"
        if activation_key not in cache:
            continue
        
        activations = cache[activation_key]  # Shape: [batch, seq_len, hidden_dim]
        
        # Extract specific token position if requested
        if token_position is not None:
            activations = activations[:, token_position, :]  # Shape: [batch, hidden_dim]
        else:
            # Average over sequence length
            activations = activations.mean(dim=1)  # Shape: [batch, hidden_dim]
        
        # Remove batch dimension and flatten
        activations = activations.squeeze(0)  # Shape: [hidden_dim]
        
        activations_dict[layer] = activations
    
    return activations_dict


# ============================================================================
# SECTION 3: METRIC CALCULATION FUNCTIONS
# ============================================================================

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """●METHOD|input:tensor_tensor|output:float|operation:compute_cosine_similarity"""
    # Normalize vectors
    a_norm = F.normalize(a, p=2, dim=0)
    b_norm = F.normalize(b, p=2, dim=0)
    # Compute cosine similarity
    similarity = torch.dot(a_norm, b_norm).item()
    return float(similarity)


def calculate_final_layer_similarity(
    layer_activations: Dict[int, torch.Tensor],
    final_layer: int
) -> Dict[int, float]:
    """●METHOD|input:dict_int|output:dict|operation:compute_similarity_to_final_layer
    
    Calculate cosine similarity between each layer and the final layer.
    
    Returns:
        Dictionary mapping layer -> similarity score
    """
    if final_layer not in layer_activations:
        return {}
    
    final_activation = layer_activations[final_layer]
    similarities = {}
    
    for layer, activation in layer_activations.items():
        if layer == final_layer:
            similarities[layer] = 1.0
        else:
            similarities[layer] = cosine_similarity(activation, final_activation)
    
    return similarities


def find_convergence_velocity(
    similarities: Dict[int, float],
    threshold: float = 0.9
) -> Optional[int]:
    """●METHOD|input:dict_float|output:int|operation:find_first_layer_above_threshold
    
    Find the earliest layer where similarity to final layer exceeds threshold.
    
    Returns:
        Layer index where convergence occurs, or None if never converges
    """
    sorted_layers = sorted(similarities.keys())
    for layer in sorted_layers:
        if similarities[layer] >= threshold:
            return layer
    return None


def calculate_representation_stability(
    layer_activations: Dict[int, torch.Tensor]
) -> Dict[int, float]:
    """●METHOD|input:dict|output:dict|operation:compute_consecutive_layer_similarity
    
    Calculate cosine similarity between consecutive layers.
    
    Returns:
        Dictionary mapping layer -> similarity to next layer
    """
    stability = {}
    sorted_layers = sorted(layer_activations.keys())
    
    for i in range(len(sorted_layers) - 1):
        layer = sorted_layers[i]
        next_layer = sorted_layers[i + 1]
        stability[layer] = cosine_similarity(
            layer_activations[layer],
            layer_activations[next_layer]
        )
    
    return stability


def find_stability_point(
    stability: Dict[int, float],
    threshold: float = 0.95,
    window: int = 3
) -> Optional[int]:
    """●METHOD|input:dict_float_int|output:int|operation:find_first_stable_window
    
    Find first layer where stability > threshold for 'window' consecutive layers.
    
    Returns:
        Layer index where stability begins, or None
    """
    sorted_layers = sorted(stability.keys())
    
    for i in range(len(sorted_layers) - window + 1):
        window_layers = sorted_layers[i:i+window]
        if all(stability.get(layer, 0.0) >= threshold for layer in window_layers):
            return window_layers[0]
    
    return None


def calculate_cross_encoding_convergence(
    nl_activations: Dict[int, torch.Tensor],
    vn_activations: Dict[int, torch.Tensor]
) -> Dict[int, float]:
    """●METHOD|input:dict_dict|output:dict|operation:compute_nl_vn_similarity_per_layer
    
    Calculate cosine similarity between NL and VN activations at each layer.
    
    Returns:
        Dictionary mapping layer -> NL-VN similarity
    """
    convergence = {}
    common_layers = set(nl_activations.keys()) & set(vn_activations.keys())
    
    for layer in sorted(common_layers):
        convergence[layer] = cosine_similarity(
            nl_activations[layer],
            vn_activations[layer]
        )
    
    return convergence


def calculate_task_alignment(
    layer_activations: Dict[int, torch.Tensor],
    task_vector: torch.Tensor
) -> Dict[int, float]:
    """●METHOD|input:dict_tensor|output:dict|operation:compute_alignment_with_task_vector
    
    Calculate cosine similarity between each layer's activation and a task vector.
    
    Args:
        layer_activations: Dictionary of layer -> activation
        task_vector: Reference task vector (e.g., average of all task activations)
        
    Returns:
        Dictionary mapping layer -> task alignment score
    """
    alignment = {}
    for layer, activation in layer_activations.items():
        alignment[layer] = cosine_similarity(activation, task_vector)
    return alignment


def compute_task_vector(
    activations_list: List[torch.Tensor]
) -> torch.Tensor:
    """●METHOD|input:list_tensor|output:tensor|operation:average_activations_to_task_vector
    
    Compute task vector as average of multiple activations.
    
    Args:
        activations_list: List of activation tensors
        
    Returns:
        Average activation vector
    """
    if len(activations_list) == 0:
        raise ValueError("Cannot compute task vector from empty list")
    
    stacked = torch.stack(activations_list, dim=0)  # Shape: [n_samples, hidden_dim]
    return stacked.mean(dim=0)  # Shape: [hidden_dim]


# ============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_convergence_curves(
    results: List[Dict[str, Any]],
    output_path: Path,
    encoding: str = "both"
) -> None:
    """●METHOD|input:list_dict_path_str|output:None|operation:plot_final_layer_similarity_curves
    
    Plot convergence curves showing final-layer similarity vs layer depth.
    
    Args:
        results: List of result dictionaries with convergence data
        output_path: Path to save figure
        encoding: "nl", "vn", or "both"
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    layers = CONFIG["layers_to_sample"]
    
    if encoding in ["nl", "both"]:
        nl_data = [r for r in results if r.get("encoding") == "nl"]
        if nl_data:
            nl_similarities = []
            nl_stds = []
            for layer in layers:
                layer_sims = [
                    r["final_layer_similarity"].get(layer, 0.0)
                    for r in nl_data
                    if layer in r.get("final_layer_similarity", {})
                ]
                if layer_sims:
                    nl_similarities.append(np.mean(layer_sims))
                    nl_stds.append(np.std(layer_sims))
                else:
                    nl_similarities.append(0.0)
                    nl_stds.append(0.0)
            
            ax.plot(layers, nl_similarities, 'o-', label='NL', linewidth=2, markersize=6, color='#e74c3c')
            if any(std > 0 for std in nl_stds):
                ax.fill_between(
                    layers, 
                    np.array(nl_similarities) - np.array(nl_stds),
                    np.array(nl_similarities) + np.array(nl_stds),
                    alpha=0.2, color='#e74c3c'
                )
    
    if encoding in ["vn", "both"]:
        vn_data = [r for r in results if r.get("encoding") == "vn"]
        if vn_data:
            vn_similarities = []
            vn_stds = []
            for layer in layers:
                layer_sims = [
                    r["final_layer_similarity"].get(layer, 0.0)
                    for r in vn_data
                    if layer in r.get("final_layer_similarity", {})
                ]
                if layer_sims:
                    vn_similarities.append(np.mean(layer_sims))
                    vn_stds.append(np.std(layer_sims))
                else:
                    vn_similarities.append(0.0)
                    vn_stds.append(0.0)
            
            ax.plot(layers, vn_similarities, 's-', label='VN', linewidth=2, markersize=6, color='#3498db')
            if any(std > 0 for std in vn_stds):
                ax.fill_between(
                    layers,
                    np.array(vn_similarities) - np.array(vn_stds),
                    np.array(vn_similarities) + np.array(vn_stds),
                    alpha=0.2, color='#3498db'
                )
    
    ax.axhline(y=CONFIG["convergence_threshold"], color='gray', linestyle='--', alpha=0.5, label=f'Convergence threshold ({CONFIG["convergence_threshold"]})')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Final-Layer Similarity', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Velocity: Final-Layer Similarity vs Layer Depth', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(layers) - 1, max(layers) + 1])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved convergence curves to: {output_path}")


def plot_velocity_histogram(
    results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """●METHOD|input:list_dict_path|output:None|operation:plot_convergence_layer_distribution
    
    Plot histogram showing distribution of convergence layers for NL vs VN.
    """
    nl_velocities = [
        r["convergence_velocity"]
        for r in results
        if r.get("encoding") == "nl" and r.get("convergence_velocity") is not None
    ]
    vn_velocities = [
        r["convergence_velocity"]
        for r in results
        if r.get("encoding") == "vn" and r.get("convergence_velocity") is not None
    ]
    
    if not nl_velocities and not vn_velocities:
        print("  Warning: No convergence velocity data to plot")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.arange(0, CONFIG["final_layer"] + 2, 2)
    
    if nl_velocities:
        ax.hist(nl_velocities, bins=bins, alpha=0.6, label=f'NL (n={len(nl_velocities)})', color='#e74c3c', edgecolor='black')
    
    if vn_velocities:
        ax.hist(vn_velocities, bins=bins, alpha=0.6, label=f'VN (n={len(vn_velocities)})', color='#3498db', edgecolor='black')
    
    ax.set_xlabel('Convergence Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Convergence Velocity Distribution: NL vs VN', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved velocity histogram to: {output_path}")


def plot_trajectory_pca(
    nl_activations: Dict[int, torch.Tensor],
    vn_activations: Dict[int, torch.Tensor],
    output_path: Path,
    case_id: str
) -> None:
    """●METHOD|input:dict_dict_path_str|output:None|operation:plot_pca_trajectory_through_layers
    
    Plot PCA projection of activations across layers showing NL vs VN trajectories.
    """
    # Collect all activations
    all_activations = []
    labels = []
    layer_indices = []
    
    for layer in sorted(set(nl_activations.keys()) | set(vn_activations.keys())):
        if layer in nl_activations:
            all_activations.append(nl_activations[layer].cpu().numpy())
            labels.append('NL')
            layer_indices.append(layer)
        
        if layer in vn_activations:
            all_activations.append(vn_activations[layer].cpu().numpy())
            labels.append('VN')
            layer_indices.append(layer)
    
    if len(all_activations) < 2:
        print(f"  Warning: Not enough activations for PCA in case {case_id}")
        return
    
    # Stack and compute PCA
    X = np.stack(all_activations, axis=0)  # Shape: [n_samples, hidden_dim]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)  # Shape: [n_samples, 2]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot trajectories
    nl_mask = np.array(labels) == 'NL'
    vn_mask = np.array(labels) == 'VN'
    
    nl_layers = np.array(layer_indices)[nl_mask]
    vn_layers = np.array(layer_indices)[vn_mask]
    nl_pca = X_pca[nl_mask]
    vn_pca = X_pca[vn_mask]
    
    if len(nl_pca) > 0:
        ax.plot(nl_pca[:, 0], nl_pca[:, 1], 'o-', label='NL', linewidth=2, markersize=8, color='#e74c3c', alpha=0.7)
        # Annotate first and last layers
        if len(nl_pca) > 0:
            ax.annotate(f'L{nl_layers[0]}', (nl_pca[0, 0], nl_pca[0, 1]), fontsize=8, color='#e74c3c')
            ax.annotate(f'L{nl_layers[-1]}', (nl_pca[-1, 0], nl_pca[-1, 1]), fontsize=8, color='#e74c3c')
    
    if len(vn_pca) > 0:
        ax.plot(vn_pca[:, 0], vn_pca[:, 1], 's-', label='VN', linewidth=2, markersize=8, color='#3498db', alpha=0.7)
        # Annotate first and last layers
        if len(vn_pca) > 0:
            ax.annotate(f'L{vn_layers[0]}', (vn_pca[0, 0], vn_pca[0, 1]), fontsize=8, color='#3498db')
            ax.annotate(f'L{vn_layers[-1]}', (vn_pca[-1, 0], vn_pca[-1, 1]), fontsize=8, color='#3498db')
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
    ax.set_title(f'Representation Trajectory: {case_id}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cross_encoding_convergence(
    results: List[Dict[str, Any]],
    output_path: Path
) -> None:
    """●METHOD|input:list_dict_path|output:None|operation:plot_nl_vn_similarity_across_layers
    
    Plot how NL and VN activations converge (or diverge) across layers.
    """
    # Get cross-encoding convergence data
    cross_data = [
        r for r in results
        if r.get("cross_encoding_convergence") is not None
    ]
    
    if not cross_data:
        print("  Warning: No cross-encoding convergence data to plot")
        return
    
    layers = CONFIG["layers_to_sample"]
    similarities = []
    stds = []
    
    for layer in layers:
        layer_sims = [
            r["cross_encoding_convergence"].get(layer, 0.0)
            for r in cross_data
            if layer in r.get("cross_encoding_convergence", {})
        ]
        if layer_sims:
            similarities.append(np.mean(layer_sims))
            stds.append(np.std(layer_sims))
        else:
            similarities.append(0.0)
            stds.append(0.0)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(layers, similarities, 'o-', linewidth=2, markersize=6, color='#9b59b6')
    if any(std > 0 for std in stds):
        ax.fill_between(
            layers,
            np.array(similarities) - np.array(stds),
            np.array(similarities) + np.array(stds),
            alpha=0.2, color='#9b59b6'
        )
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('NL-VN Similarity', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Encoding Convergence: Do NL and VN Converge to Same Representation?', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([min(layers) - 1, max(layers) + 1])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved cross-encoding convergence plot to: {output_path}")


# ============================================================================
# SECTION 5: MAIN EXPERIMENT
# ============================================================================

def run_experiment() -> None:
    """●METHOD|input:None|output:None|operation:execute_convergence_velocity_experiment
    
    Main entry point for convergence velocity experiment.
    """
    experiment_start_time = datetime.now().isoformat()
    
    # Initialize storage
    specimen_path = Path(__file__).parent
    storage = SpecimenStorage(specimen_path)
    figures_path = specimen_path / "strata" / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)
    
    print(f"●PROCESS|operation:convergence_velocity_experiment|phase:starting")
    print(f"  Storage initialized at: {specimen_path}")
    
    print("\n" + "="*80)
    print("EXPERIMENT H: CONVERGENCE VELOCITY TEST")
    print("Testing VN convergence speed vs NL across layers")
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
    
    # Note: We don't need SAE for this experiment - we're working with raw activations
    
    # Select test cases
    print("\n" + "="*80)
    print("SELECTING TEST CASES")
    print("="*80)
    
    selected_cases = []
    for complexity in ["simple", "medium", "complex"]:
        complexity_cases = get_test_cases_by_complexity(complexity)
        case_ids = list(complexity_cases.keys())
        random.shuffle(case_ids)
        sampled = case_ids[:CONFIG["test_cases_per_complexity"]]
        selected_cases.extend([(case_id, complexity_cases[case_id]) for case_id in sampled])
    
    print(f"\nSelected {len(selected_cases)} test cases:")
    print(f"  Simple: {sum(1 for _, c in selected_cases if c['complexity'] == 'simple')}")
    print(f"  Medium: {sum(1 for _, c in selected_cases if c['complexity'] == 'medium')}")
    print(f"  Complex: {sum(1 for _, c in selected_cases if c['complexity'] == 'complex')}")
    
    # Compute task vectors (average activations per category)
    print("\n" + "="*80)
    print("COMPUTING TASK VECTORS")
    print("="*80)
    
    task_vectors = {}
    categories = set(c['category'] for _, c in selected_cases)
    
    for category in categories:
        category_cases = [(cid, c) for cid, c in selected_cases if c['category'] == category]
        # Sample a few cases to compute task vector
        sample_size = min(5, len(category_cases))
        sampled = random.sample(category_cases, sample_size)
        
        activations_list = []
        for case_id, case_data in sampled:
            try:
                nl_activations = extract_multi_layer_activations(
                    engine, case_data['nl'], [CONFIG["final_layer"]], CONFIG["token_position"]
                )
                if CONFIG["final_layer"] in nl_activations:
                    activations_list.append(nl_activations[CONFIG["final_layer"]])
            except Exception as e:
                print(f"  Warning: Could not extract activation for {case_id}: {e}")
                continue
        
        if activations_list:
            task_vectors[category] = compute_task_vector(activations_list)
            print(f"  ✓ Computed task vector for category: {category} (n={len(activations_list)} samples)")
    
    # Run experiments
    print("\n" + "="*80)
    print("RUNNING CONVERGENCE VELOCITY EXPERIMENTS")
    print("="*80)
    print(f"Layers to sample: {CONFIG['layers_to_sample']}")
    print(f"Total runs: {len(selected_cases)} cases × 2 (NL/VN)")
    
    results = []
    processed = 0
    
    for case_id, case_data in selected_cases:
        print(f"\n[{processed // 2 + 1}/{len(selected_cases)}] Processing: {case_id}")
        print(f"  Complexity: {case_data['complexity']} | Category: {case_data['category']}")
        
        # Process NL
        nl_activations = None
        try:
            nl_activations = extract_multi_layer_activations(
                engine, case_data['nl'], CONFIG["layers_to_sample"], CONFIG["token_position"]
            )
            
            nl_final_similarity = calculate_final_layer_similarity(nl_activations, CONFIG["final_layer"])
            nl_velocity = find_convergence_velocity(nl_final_similarity, CONFIG["convergence_threshold"])
            nl_stability = calculate_representation_stability(nl_activations)
            nl_stability_point = find_stability_point(nl_stability, CONFIG["stability_threshold"], CONFIG["stability_window"])
            
            # Task alignment
            nl_task_alignment = {}
            if case_data['category'] in task_vectors:
                nl_task_alignment = calculate_task_alignment(nl_activations, task_vectors[case_data['category']])
            
            nl_metrics = {
                "case_id": case_id,
                "complexity": case_data['complexity'],
                "category": case_data['category'],
                "encoding": "nl",
                "convergence_velocity": nl_velocity,
                "stability_point": nl_stability_point,
                "final_layer_similarity": {k: float(v) for k, v in nl_final_similarity.items()},
                "representation_stability": {k: float(v) for k, v in nl_stability.items()},
                "task_alignment": {k: float(v) for k, v in nl_task_alignment.items()},
            }
            results.append(nl_metrics)
            processed += 1
            
        except Exception as e:
            print(f"  ✗ Error processing NL: {e}")
        
        # Process VN
        try:
            vn_activations = extract_multi_layer_activations(
                engine, case_data['vn'], CONFIG["layers_to_sample"], CONFIG["token_position"]
            )
            
            vn_final_similarity = calculate_final_layer_similarity(vn_activations, CONFIG["final_layer"])
            vn_velocity = find_convergence_velocity(vn_final_similarity, CONFIG["convergence_threshold"])
            vn_stability = calculate_representation_stability(vn_activations)
            vn_stability_point = find_stability_point(vn_stability, CONFIG["stability_threshold"], CONFIG["stability_window"])
            
            # Task alignment
            vn_task_alignment = {}
            if case_data['category'] in task_vectors:
                vn_task_alignment = calculate_task_alignment(vn_activations, task_vectors[case_data['category']])
            
            # Cross-encoding convergence (only if NL was successfully processed)
            cross_convergence = {}
            if nl_activations is not None:
                cross_convergence = calculate_cross_encoding_convergence(nl_activations, vn_activations)
            
            vn_metrics = {
                "case_id": case_id,
                "complexity": case_data['complexity'],
                "category": case_data['category'],
                "encoding": "vn",
                "convergence_velocity": vn_velocity,
                "stability_point": vn_stability_point,
                "final_layer_similarity": {k: float(v) for k, v in vn_final_similarity.items()},
                "representation_stability": {k: float(v) for k, v in vn_stability.items()},
                "task_alignment": {k: float(v) for k, v in vn_task_alignment.items()},
                "cross_encoding_convergence": {k: float(v) for k, v in cross_convergence.items()},
            }
            results.append(vn_metrics)
            processed += 1
            
            # Save trajectory plot for this case (only if NL was successfully processed)
            if nl_activations is not None:
                plot_trajectory_pca(
                    nl_activations,
                    vn_activations,
                    figures_path / f"trajectory_{case_id}.png",
                    case_id
                )
            
        except Exception as e:
            print(f"  ✗ Error processing VN: {e}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    if len(results) == 0:
        print("✗ No results to save!")
        return
    
    # Flatten nested dictionaries for Parquet storage
    flattened_results = []
    for r in results:
        flat = {
            "case_id": r["case_id"],
            "complexity": r["complexity"],
            "category": r["category"],
            "encoding": r["encoding"],
            "convergence_velocity": r.get("convergence_velocity"),
            "stability_point": r.get("stability_point"),
        }
        
        # Flatten final_layer_similarity
        for layer in CONFIG["layers_to_sample"]:
            flat[f"final_sim_L{layer}"] = r.get("final_layer_similarity", {}).get(layer)
        
        # Flatten representation_stability (only store first few layers)
        for i, layer in enumerate(CONFIG["layers_to_sample"][:-1]):
            flat[f"stability_L{layer}"] = r.get("representation_stability", {}).get(layer)
        
        # Flatten task_alignment
        for layer in CONFIG["layers_to_sample"]:
            flat[f"task_align_L{layer}"] = r.get("task_alignment", {}).get(layer)
        
        # Flatten cross_encoding_convergence (VN only)
        if "cross_encoding_convergence" in r:
            for layer in CONFIG["layers_to_sample"]:
                flat[f"cross_sim_L{layer}"] = r.get("cross_encoding_convergence", {}).get(layer)
        
        flattened_results.append(flat)
    
    # Convert to columnar format for Parquet
    metrics_dict = {}
    for key in flattened_results[0].keys():
        metrics_dict[key] = [r[key] for r in flattened_results]
    
    storage.write_metrics(metrics_dict)
    print(f"✓ Saved {len(results)} result rows to metrics.parquet")
    
    # Create visualizations
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    plot_convergence_curves(results, figures_path / "convergence_curves.png", encoding="both")
    plot_velocity_histogram(results, figures_path / "velocity_histogram.png")
    plot_cross_encoding_convergence(results, figures_path / "cross_encoding_convergence.png")
    
    # Calculate summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    nl_results = [r for r in results if r['encoding'] == 'nl']
    vn_results = [r for r in results if r['encoding'] == 'vn']
    
    nl_velocities = [r['convergence_velocity'] for r in nl_results if r.get('convergence_velocity') is not None]
    vn_velocities = [r['convergence_velocity'] for r in vn_results if r.get('convergence_velocity') is not None]
    
    if nl_velocities:
        print(f"\nNL Convergence Velocity:")
        print(f"  Mean: {np.mean(nl_velocities):.1f} layers")
        print(f"  Median: {np.median(nl_velocities):.1f} layers")
        print(f"  Std: {np.std(nl_velocities):.1f} layers")
        print(f"  Converged: {len(nl_velocities)}/{len(nl_results)} cases")
    
    if vn_velocities:
        print(f"\nVN Convergence Velocity:")
        print(f"  Mean: {np.mean(vn_velocities):.1f} layers")
        print(f"  Median: {np.median(vn_velocities):.1f} layers")
        print(f"  Std: {np.std(vn_velocities):.1f} layers")
        print(f"  Converged: {len(vn_velocities)}/{len(vn_results)} cases")
    
    if nl_velocities and vn_velocities:
        mean_diff = np.mean(nl_velocities) - np.mean(vn_velocities)
        print(f"\nVN Advantage: {mean_diff:.1f} layers earlier convergence on average")
    
    # Update manifest
    manifest = {
        "specimen_id": "2024_12_20_vn_convergence_velocity",
        "created": experiment_start_time,
        "completed": datetime.now().isoformat(),
        "taxonomy": {
            "domain": "interpretability",
            "method": "layer_analysis"
        },
        "tags": ["vn", "convergence", "velocity", "gemma-2-2b", "multi_layer"],
        "config": CONFIG,
        "summary": {
            "total_cases": len(selected_cases),
            "total_runs": len(results),
            "nl_converged": len(nl_velocities),
            "vn_converged": len(vn_velocities),
            "nl_mean_velocity": float(np.mean(nl_velocities)) if nl_velocities else None,
            "vn_mean_velocity": float(np.mean(vn_velocities)) if vn_velocities else None,
        }
    }
    
    storage.write_manifest(manifest)
    print(f"\n✓ Experiment complete!")
    print(f"  Results saved to: {specimen_path / 'strata' / 'metrics.parquet'}")
    print(f"  Figures saved to: {figures_path}")


if __name__ == "__main__":
    run_experiment()

