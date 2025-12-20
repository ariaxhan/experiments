"""●COMPONENT|Ψ:interference_engine|Ω:spectral_analysis_purity_drift_absorption

InterferenceEngine: The Detector - Mathematical analysis of spectra.
Ported from universal-spectroscopy-engine.

This component implements the three core hypotheses:
- H1: Spectral Purity (Hallucination Detection)
- H2: Doppler Shift (Semantic Drift)
- H3: Absorption (Model Blindness)
"""

from typing import List
import torch
import torch.nn.functional as F

from .spectrum import Spectrum


class InterferenceEngine:
    """
    The Detector: Mathematical analysis module.
    
    Calculates:
    - Spectral Purity (Signal-to-Noise) - H1
    - Doppler Shift (Semantic Drift) - H2
    - Absorption (Missing Features) - H3
    """
    
    def __init__(self, purity_threshold: float = 0.1, drift_threshold: float = 0.5):
        """
        Initialize InterferenceEngine.
        
        Args:
            purity_threshold: Threshold for considering features as "signal" vs "noise"
            drift_threshold: Threshold for significant drift detection
        """
        self.purity_threshold = purity_threshold
        self.drift_threshold = drift_threshold
    
    def calculate_purity(self, spectrum: Spectrum) -> float:
        """
        Calculate spectral purity (H1: Spectral Purity Hypothesis).
        
        Hallucinations manifest as low spectral purity (high entropy/noise,
        few distinct peaks).
        
        Purity is calculated as:
        - Signal-to-noise ratio: ratio of high-intensity features to low-intensity features
        - Entropy measure: lower entropy = higher purity (more concentrated activations)
        
        Args:
            spectrum: Input spectrum
            
        Returns:
            Purity score between 0.0 (low purity, possible hallucination) and 1.0 (high purity)
        """
        if len(spectrum) == 0:
            return 0.0
        
        intensities = spectrum.intensities
        
        # Normalize intensities
        total_intensity = intensities.sum()
        if total_intensity == 0:
            return 0.0
        
        normalized = intensities / total_intensity
        
        # Calculate entropy (Shannon entropy)
        # Higher entropy = more spread out = lower purity
        # Lower entropy = more concentrated = higher purity
        entropy = -(normalized * torch.log(normalized + 1e-10)).sum()
        max_entropy = torch.log(torch.tensor(len(intensities), dtype=torch.float32))
        normalized_entropy = entropy / max_entropy
        
        # Purity is inverse of normalized entropy
        entropy_purity = 1.0 - normalized_entropy.item()
        
        # Signal-to-noise ratio
        # Signal = features above threshold
        # Noise = features below threshold
        signal_mask = intensities >= (intensities.max() * self.purity_threshold)
        signal_power = intensities[signal_mask].sum()
        noise_power = intensities[~signal_mask].sum()
        
        if noise_power == 0:
            snr_purity = 1.0
        else:
            snr = signal_power / noise_power
            # Normalize SNR to [0, 1] range (assuming max reasonable SNR ~100)
            snr_purity = min(1.0, snr.item() / 100.0)
        
        # Combine entropy and SNR measures (weighted average)
        purity = 0.6 * entropy_purity + 0.4 * snr_purity
        
        return max(0.0, min(1.0, purity))
    
    def calculate_drift(
        self,
        spec_a: Spectrum,
        spec_b: Spectrum,
        top_k: int = 100
    ) -> float:
        """
        Calculate semantic drift between two spectra (H2: Doppler Shift Hypothesis).
        
        As information passes through agent chains, semantic meaning "redshifts"
        (generalizes) or "blueshifts" (distorts).
        
        Uses intersection over union (IoU) or cosine similarity of top-k features.
        
        Args:
            spec_a: First spectrum (e.g., input)
            spec_b: Second spectrum (e.g., output)
            top_k: Number of top features to compare
            
        Returns:
            Drift magnitude between 0.0 (no drift) and 1.0 (complete drift)
        """
        if len(spec_a) == 0 or len(spec_b) == 0:
            return 1.0  # Complete drift if one spectrum is empty
        
        # Get top-k features from each spectrum
        top_a = set(spec_a.get_top_features(k=top_k).cpu().numpy().tolist())
        top_b = set(spec_b.get_top_features(k=top_k).cpu().numpy().tolist())
        
        # Calculate intersection over union
        intersection = len(top_a & top_b)
        union = len(top_a | top_b)
        
        if union == 0:
            return 1.0
        
        iou = intersection / union
        
        # Drift is inverse of IoU
        drift = 1.0 - iou
        
        # Also calculate cosine similarity of intensity vectors
        # Create feature vectors for top features
        all_features = sorted(list(top_a | top_b))
        feature_to_idx = {f: i for i, f in enumerate(all_features)}
        
        # Build intensity vectors
        vec_a = torch.zeros(len(all_features))
        vec_b = torch.zeros(len(all_features))
        
        for i, feature in enumerate(spec_a.wavelengths):
            if feature.item() in feature_to_idx:
                vec_a[feature_to_idx[feature.item()]] = spec_a.intensities[i]
        
        for i, feature in enumerate(spec_b.wavelengths):
            if feature.item() in feature_to_idx:
                vec_b[feature_to_idx[feature.item()]] = spec_b.intensities[i]
        
        # Normalize vectors
        vec_a_norm = F.normalize(vec_a, p=2, dim=0)
        vec_b_norm = F.normalize(vec_b, p=2, dim=0)
        
        # Cosine similarity
        cosine_sim = (vec_a_norm * vec_b_norm).sum().item()
        cosine_drift = 1.0 - cosine_sim
        
        # Combine IoU and cosine similarity measures
        drift = 0.5 * drift + 0.5 * cosine_drift
        
        return max(0.0, min(1.0, drift))
    
    def detect_absorption(
        self,
        input_spec: Spectrum,
        output_spec: Spectrum,
        threshold: float = 0.5
    ) -> List[int]:
        """
        Detect features present in input but absent in output (H3: Absorption Hypothesis).
        
        If a model ignores an instruction (e.g., safety filter), the relevant
        concept feature will be absent in the output spectrum despite being
        present in the input.
        
        Args:
            input_spec: Input spectrum
            output_spec: Output spectrum to compare against
            threshold: Minimum intensity in input to consider a feature as "present"
            
        Returns:
            List of feature indices (wavelengths) that were absorbed
            (present in input, absent or significantly reduced in output)
        """
        if len(input_spec) == 0:
            return []
        
        # Get features present in input (above threshold)
        input_mask = input_spec.intensities >= threshold
        input_features = set(
            input_spec.wavelengths[input_mask].cpu().numpy().tolist()
        )
        
        if len(input_features) == 0:
            return []
        
        # Get features present in output
        output_features = set(
            output_spec.wavelengths.cpu().numpy().tolist()
        )
        
        # Find features in input but not in output (or significantly reduced)
        absorbed = []
        
        for feature_idx in input_features:
            # Check if feature is in output
            if feature_idx not in output_features:
                absorbed.append(int(feature_idx))
            else:
                # Feature exists but check if intensity dropped significantly
                input_intensity = input_spec.intensities[
                    input_spec.wavelengths == feature_idx
                ].max()
                
                output_intensity = output_spec.intensities[
                    output_spec.wavelengths == feature_idx
                ].max()
                
                # If output intensity is less than 50% of input, consider absorbed
                if output_intensity < (input_intensity * 0.5):
                    absorbed.append(int(feature_idx))
        
        return absorbed

