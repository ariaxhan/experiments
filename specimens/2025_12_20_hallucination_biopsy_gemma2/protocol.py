"""●COMPONENT|Ψ:hallucination_biopsy_protocol|Ω:detect_hallucination_biomarkers_via_sae

This protocol performs differential diagnosis on language model outputs to identify
hallucination signatures using Sparse Autoencoder (SAE) analysis.

Methodology:
1. Load Gemma-2-2b model and GemmaScope SAE (layer 5, 16k features)
2. For each experiment pair (fact vs hallucination):
   - Extract feature activations from both texts
   - Identify features unique to hallucination
   - Decode feature meanings via vocabulary projection
3. Store results in Specimen Vault format:
   - Metrics: Comparative statistics (Parquet)
   - Tensors: Full activation patterns (Zarr)

Dependencies: transformer_lens, sae_lens, torch, numpy
"""

from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import numpy as np
from sae_lens import SAE
from transformer_lens import HookedTransformer

from protocols.storage import SpecimenStorage


# ============================================================================
# SECTION 1: INSTRUMENT SETUP
# ============================================================================

def initialize_instruments(device: str = None) -> Tuple[HookedTransformer, SAE, str, Dict]:
    """●METHOD|input:str|output:tuple|operation:load_model_sae_return_metadata
    
    Load the language model and SAE analyzer.
    
    Args:
        device: Device to use ('mps', 'cuda', or 'cpu'). Auto-detects if None.
    
    Returns:
        Tuple of (model, sae, device, setup_metadata)
    """
    # Auto-detect device
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"●PROCESS|operation:loading_instruments|device:{device}")
    
    # Load SAE (the "microscope")
    print("  Loading SAE microscope...")
    sae_release = "gemma-scope-2b-pt-res-canonical"
    sae_id = "layer_5/width_16k/canonical"
    sae = SAE.from_pretrained(
        release=sae_release,
        sae_id=sae_id,
        device=device
    )
    
    # Load model (the "subject")
    print("  Loading Gemma-2-2b model...")
    model_name = "gemma-2-2b"
    model = HookedTransformer.from_pretrained(model_name, device=device)
    
    # Collect metadata
    setup_metadata = {
        "device": device,
        "model_name": model_name,
        "model_config": {
            "n_layers": model.cfg.n_layers,
            "d_model": model.cfg.d_model,
            "d_vocab": model.cfg.d_vocab,
            "n_heads": model.cfg.n_heads,
        },
        "sae_release": sae_release,
        "sae_id": sae_id,
        "sae_config": {
            "d_in": sae.cfg.d_in,
            "d_sae": sae.cfg.d_sae,
            "hook_name": "blocks.5.hook_resid_post",
        },
    }
    
    print("  ✓ Instruments ready")
    return model, sae, device, setup_metadata


# ============================================================================
# SECTION 2: DIAGNOSTIC PROCEDURES
# ============================================================================

def take_biopsy(text: str, model: HookedTransformer, sae: SAE) -> Dict:
    """●METHOD|input:str_model_sae|output:dict|operation:extract_feature_signature
    
    Extract SAE feature activations from text (the "biopsy").
    
    Args:
        text: Input text to analyze
        model: Language model
        sae: Sparse autoencoder
    
    Returns:
        Dictionary with indices, magnitudes, counts, and energy
    """
    # Run model to get activations
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens)
    
    # Extract activation from last token (where prediction happens)
    activations = cache["blocks.5.hook_resid_post"][0, -1, :]
    
    # Apply SAE to get feature activations
    activations = activations.unsqueeze(0)
    feature_acts = sae.encode(activations).squeeze()
    
    # Filter for active features
    active_indices = torch.nonzero(feature_acts > 0).squeeze()
    if active_indices.dim() == 0:  # Handle single element case
        active_indices = active_indices.unsqueeze(0)
    magnitudes = feature_acts[active_indices]
    
    return {
        "indices": active_indices.tolist() if len(active_indices) > 0 else [],
        "magnitudes": magnitudes.tolist() if len(magnitudes) > 0 else [],
        "total_active": len(active_indices),
        "energy": magnitudes.sum().item() if len(magnitudes) > 0 else 0.0
    }


def get_loudest_unique_features(
    fact_text: str, 
    hall_text: str, 
    model: HookedTransformer, 
    sae: SAE,
    top_k: int = 5
) -> List[int]:
    """●METHOD|input:str_str_model_sae_int|output:list|operation:find_strongest_hallucination_features
    
    Find features that are active in hallucination but not in fact, sorted by magnitude.
    
    Args:
        fact_text: Ground truth text
        hall_text: Hallucinated text
        model: Language model
        sae: Sparse autoencoder
        top_k: Number of top features to return
    
    Returns:
        List of feature indices (loudest first)
    """
    # Get activations for both texts
    tokens_fact = model.to_tokens(fact_text)
    _, cache_fact = model.run_with_cache(tokens_fact)
    act_fact = cache_fact["blocks.5.hook_resid_post"][0, -1, :]
    feat_fact = sae.encode(act_fact.unsqueeze(0)).squeeze()
    
    tokens_hall = model.to_tokens(hall_text)
    _, cache_hall = model.run_with_cache(tokens_hall)
    act_hall = cache_hall["blocks.5.hook_resid_post"][0, -1, :]
    feat_hall = sae.encode(act_hall.unsqueeze(0)).squeeze()
    
    # Find unique features (active in hallucination, zero in fact)
    hall_active = (feat_hall > 0)
    fact_inactive = (feat_fact == 0)
    unique_mask = hall_active & fact_inactive
    
    unique_indices = torch.nonzero(unique_mask).squeeze()
    if unique_indices.dim() == 0:
        unique_indices = unique_indices.unsqueeze(0)
    
    if len(unique_indices) == 0:
        return []
    
    unique_magnitudes = feat_hall[unique_indices]
    
    # Sort by magnitude (loudest first)
    sorted_indices = unique_indices[torch.argsort(unique_magnitudes, descending=True)]
    
    return sorted_indices[:top_k].tolist()


def translate_feature(feature_id: int, model: HookedTransformer, sae: SAE, top_k: int = 5) -> Dict:
    """●METHOD|input:int_model_sae_int|output:dict|operation:decode_feature_via_vocabulary
    
    Translate a feature ID into the words it promotes.
    
    Args:
        feature_id: SAE feature index
        model: Language model
        sae: Sparse autoencoder
        top_k: Number of top words to return
    
    Returns:
        Dictionary with words, logits, and feature_id
    """
    # Get feature direction in model space
    feature_direction = sae.W_dec[feature_id]
    
    # Project to vocabulary
    logits = model.unembed(feature_direction)
    
    # Get top words
    top_token_ids = logits.argsort(descending=True)[:top_k]
    top_words = model.to_str_tokens(top_token_ids)
    top_logits = logits[top_token_ids].tolist()
    
    return {
        "feature_id": feature_id,
        "words": top_words,
        "logits": top_logits,
    }


def run_differential_diagnosis(
    fact_text: str, 
    hall_text: str, 
    model: HookedTransformer, 
    sae: SAE
) -> Dict:
    """●METHOD|input:str_str_model_sae|output:dict|operation:compare_fact_vs_hallucination
    
    Compare feature activations between fact and hallucination.
    
    Args:
        fact_text: Ground truth text
        hall_text: Hallucinated text
        model: Language model
        sae: Sparse autoencoder
    
    Returns:
        Diagnosis dictionary with comparative metrics and biomarkers
    """
    # Get signatures
    sig_fact = take_biopsy(fact_text, model, sae)
    sig_hall = take_biopsy(hall_text, model, sae)
    
    # Compare
    set_fact = set(sig_fact["indices"])
    set_hall = set(sig_hall["indices"])
    
    unique_to_hallucination = list(set_hall - set_fact)
    missing_from_hallucination = list(set_fact - set_hall)
    
    return {
        "spectral_metrics": {
            "control_entropy": sig_fact["total_active"],
            "sample_entropy": sig_hall["total_active"],
            "energy_diff": sig_hall["energy"] - sig_fact["energy"]
        },
        "biomarkers": {
            "unique_to_hallucination_count": len(unique_to_hallucination),
            "missing_grounding_count": len(missing_from_hallucination),
        },
        "signatures": {
            "fact": sig_fact,
            "hallucination": sig_hall,
        }
    }


# ============================================================================
# SECTION 3: BATCH ANALYSIS
# ============================================================================

def analyze_batch(
    experiments: List[Dict], 
    model: HookedTransformer, 
    sae: SAE
) -> Tuple[Dict[str, List], Dict[str, np.ndarray]]:
    """●METHOD|input:list_model_sae|output:tuple|operation:process_multiple_experiments
    
    Run differential diagnosis on multiple fact/hallucination pairs.
    
    Args:
        experiments: List of dicts with 'name', 'fact', 'hallucination' keys
        model: Language model
        sae: Sparse autoencoder
    
    Returns:
        Tuple of (metrics_dict, tensors_dict) ready for SpecimenStorage
    """
    print(f"●PROCESS|operation:batch_analysis|count:{len(experiments)}")
    
    # Initialize storage structures
    metrics = {
        "experiment_index": [],
        "experiment_name": [],
        "fact_text": [],
        "hallucination_text": [],
        "fact_total_active": [],
        "hall_total_active": [],
        "fact_energy": [],
        "hall_energy": [],
        "energy_diff": [],
        "unique_to_hall_count": [],
        "missing_from_hall_count": [],
        "top_feature_1": [],
        "top_feature_2": [],
        "top_feature_3": [],
        "top_feature_4": [],
        "top_feature_5": [],
        "top_feature_1_words": [],
        "top_feature_2_words": [],
        "top_feature_3_words": [],
        "top_feature_4_words": [],
        "top_feature_5_words": [],
    }
    
    # Collect full activation tensors
    d_sae = sae.cfg.d_sae
    fact_activations = np.zeros((len(experiments), d_sae), dtype=np.float32)
    hall_activations = np.zeros((len(experiments), d_sae), dtype=np.float32)
    
    for i, exp in enumerate(experiments):
        print(f"\n  Experiment {i+1}/{len(experiments)}: {exp['name']}")
        
        # Run diagnosis
        diagnosis = run_differential_diagnosis(exp['fact'], exp['hallucination'], model, sae)
        
        # Get loudest unique features
        loudest_indices = get_loudest_unique_features(exp['fact'], exp['hallucination'], model, sae, top_k=5)
        
        # Translate features
        translations = []
        for feat_id in loudest_indices:
            trans = translate_feature(feat_id, model, sae, top_k=3)
            translations.append(trans)
        
        # Store metrics
        metrics["experiment_index"].append(i + 1)
        metrics["experiment_name"].append(exp['name'])
        metrics["fact_text"].append(exp['fact'])
        metrics["hallucination_text"].append(exp['hallucination'])
        metrics["fact_total_active"].append(diagnosis['signatures']['fact']['total_active'])
        metrics["hall_total_active"].append(diagnosis['signatures']['hallucination']['total_active'])
        metrics["fact_energy"].append(diagnosis['signatures']['fact']['energy'])
        metrics["hall_energy"].append(diagnosis['signatures']['hallucination']['energy'])
        metrics["energy_diff"].append(diagnosis['spectral_metrics']['energy_diff'])
        metrics["unique_to_hall_count"].append(diagnosis['biomarkers']['unique_to_hallucination_count'])
        metrics["missing_from_hall_count"].append(diagnosis['biomarkers']['missing_grounding_count'])
        
        # Store top 5 features (pad with -1 if fewer than 5)
        for j in range(5):
            feat_key = f"top_feature_{j+1}"
            words_key = f"top_feature_{j+1}_words"
            if j < len(loudest_indices):
                metrics[feat_key].append(loudest_indices[j])
                metrics[words_key].append(", ".join(translations[j]['words']))
            else:
                metrics[feat_key].append(-1)
                metrics[words_key].append("")
        
        # Store full activations for tensor storage
        # Re-extract to get full vectors (not just active indices)
        tokens_fact = model.to_tokens(exp['fact'])
        _, cache_fact = model.run_with_cache(tokens_fact)
        act_fact = cache_fact["blocks.5.hook_resid_post"][0, -1, :]
        feat_fact = sae.encode(act_fact.unsqueeze(0)).squeeze()
        fact_activations[i, :] = feat_fact.detach().cpu().numpy()
        
        tokens_hall = model.to_tokens(exp['hallucination'])
        _, cache_hall = model.run_with_cache(tokens_hall)
        act_hall = cache_hall["blocks.5.hook_resid_post"][0, -1, :]
        feat_hall = sae.encode(act_hall.unsqueeze(0)).squeeze()
        hall_activations[i, :] = feat_hall.detach().cpu().numpy()
        
        print(f"    ✓ Unique features: {len(loudest_indices)}")
        if loudest_indices:
            print(f"    Top feature: #{loudest_indices[0]} → {translations[0]['words']}")
    
    tensors = {
        "fact_activations": fact_activations,
        "hall_activations": hall_activations,
    }
    
    return metrics, tensors


# ============================================================================
# SECTION 4: MAIN EXPERIMENT
# ============================================================================

def run_experiment() -> None:
    """●METHOD|input:None|output:None|operation:execute_hallucination_biopsy_experiment
    
    Main entry point for the hallucination biopsy experiment.
    
    Workflow:
    1. Initialize instruments (model + SAE)
    2. Define experiment pairs (fact vs hallucination)
    3. Run batch analysis
    4. Save results to Specimen Vault (Parquet + Zarr)
    """
    experiment_start_time = datetime.now().isoformat()
    
    # Initialize storage
    specimen_path = Path(__file__).parent
    storage = SpecimenStorage(specimen_path)
    print(f"●PROCESS|operation:hallucination_biopsy|phase:starting")
    print(f"  Storage initialized at: {specimen_path}")
    
    # Load instruments
    model, sae, device, setup_metadata = initialize_instruments()
    
    # Define experiments
    experiments = [
        {
            "name": "Geography Teleportation",
            "fact": "The Eiffel Tower is located in Paris",
            "hallucination": "The Eiffel Tower is located in Rome"
        },
        {
            "name": "Geography Teleportation 2",
            "fact": "The Golden Gate Bridge is in San Francisco",
            "hallucination": "The Golden Gate Bridge is in New York"
        },
        {
            "name": "Historical Anachronism",
            "fact": "William Shakespeare wrote Hamlet",
            "hallucination": "William Shakespeare wrote Star Wars"
        },
        {
            "name": "Biological Impossibility",
            "fact": "Dogs walk on four legs",
            "hallucination": "Dogs fly with two wings"
        },
        {
            "name": "Mathematical Inversion",
            "fact": "Five is greater than two",
            "hallucination": "Five is smaller than two"
        }
    ]
    
    print(f"\n●PROCESS|operation:defining_experiments|count:{len(experiments)}")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp['name']}")
    
    # Run batch analysis
    metrics, tensors = analyze_batch(experiments, model, sae)
    
    # Save results
    print(f"\n●PROCESS|operation:saving_results|phase:execution")
    
    print("  Writing metrics to Parquet...")
    storage.write_metrics(metrics)
    print(f"    ✓ Saved {len(metrics['experiment_index'])} experiment records")
    
    print("  Writing activation tensors to Zarr...")
    storage.write_tensors("fact_activations", tensors["fact_activations"])
    storage.write_tensors("hall_activations", tensors["hall_activations"])
    print(f"    ✓ Saved fact_activations: {tensors['fact_activations'].shape}")
    print(f"    ✓ Saved hall_activations: {tensors['hall_activations'].shape}")
    
    # Save experiment metadata
    experiment_end_time = datetime.now().isoformat()
    metadata = {
        "experiment_type": "hallucination_biopsy_batch_analysis",
        "setup": setup_metadata,
        "experiment_config": {
            "total_experiments": len(experiments),
            "experiment_names": [exp['name'] for exp in experiments],
        },
        "timing": {
            "start": experiment_start_time,
            "end": experiment_end_time,
        }
    }
    storage.write_manifest(metadata)
    print("    ✓ Saved experiment metadata")
    
    print("\n✓ Hallucination biopsy complete!")
    print("  Next steps:")
    print("    1. Run: python scripts/index_vault.py")
    print("    2. Query results via VaultQuery")
    print(f"    3. Inspect tensors: storage.read_tensor_lazy('fact_activations')")


if __name__ == "__main__":
    run_experiment()
