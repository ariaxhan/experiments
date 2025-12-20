#!/usr/bin/env python3
"""●COMPONENT|Ψ:quick_stats_wrapper|Ω:backward_compatible_results_viewer

Quick Stats - Backward Compatible Wrapper

This is a convenience wrapper around view_results.py for backward compatibility.
Can be placed in specimen directories or run from anywhere.

Usage:
    # From specimen directory
    python quick_stats.py
    
    # From anywhere
    python scripts/quick_stats.py specimens/2024_12_20_vn_noise_robustness
"""

import sys
from pathlib import Path

# Add parent directory to path
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

# Import and use the universal viewer
from scripts.view_results import view_specimen

if __name__ == "__main__":
    # If run from specimen directory, use current directory
    # Otherwise, expect specimen path as argument
    if len(sys.argv) > 1:
        specimen_path = Path(sys.argv[1])
    else:
        # Assume we're in a specimen directory
        specimen_path = Path.cwd()
    
    view_specimen(specimen_path)

