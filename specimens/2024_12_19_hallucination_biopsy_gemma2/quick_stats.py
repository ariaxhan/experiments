#!/usr/bin/env python3
"""Quick statistics viewer - one-liner friendly"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from protocols.storage import SpecimenStorage

# Load data
storage = SpecimenStorage(Path(__file__).parent)
metrics = storage.read_metrics()

# Print compact summary
print(f"\n{'='*60}")
print(f"QUICK STATS: {len(metrics)} experiments")
print(f"{'='*60}")

for row in metrics.iter_rows(named=True):
    print(f"\n{row['experiment_name']}:")
    print(f"  Unique features: {row['unique_to_hall_count']}")
    print(f"  Energy diff: {row['energy_diff']:+.3f}")
    print(f"  Top feature: #{row['top_feature_1']} → {row['top_feature_1_words']}")

print(f"\n{'='*60}")
print(f"Averages:")
print(f"  Unique features: {metrics['unique_to_hall_count'].mean():.1f}")
print(f"  Energy diff: {metrics['energy_diff'].mean():.3f}")
print(f"{'='*60}\n")

