#!/usr/bin/env python3
"""●COMPONENT|Ψ:migration_tool|Ω:migrate_legacy_specimens_to_runs_structure

Migrates existing specimens from old structure (strata/ directly) to new runs structure.

Usage:
    python scripts/migrate_to_runs.py [specimen_path]
    
If no path provided, migrates all specimens in specimens/ directory.
"""

import sys
from pathlib import Path
import shutil
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protocols.storage import SpecimenStorage


def migrate_specimen(specimen_path: Path) -> bool:
    """Migrate a single specimen from old to new structure."""
    specimen_path = Path(specimen_path)
    
    if not specimen_path.exists():
        print(f"✗ Specimen not found: {specimen_path}")
        return False
    
    old_strata = specimen_path / "strata"
    if not old_strata.exists():
        print(f"✓ {specimen_path.name}: No legacy strata directory, already migrated or empty")
        return True
    
    # Check if already has runs directory
    runs_dir = specimen_path / "runs"
    if runs_dir.exists() and any(runs_dir.iterdir()):
        print(f"⚠ {specimen_path.name}: Already has runs directory, skipping")
        return False
    
    # Create runs directory
    runs_dir.mkdir(exist_ok=True)
    
    # Check for existing metrics/manifest
    old_metrics = old_strata / "metrics.parquet"
    old_manifest = old_strata / "manifest.msgpack"
    
    if not old_metrics.exists() and not old_manifest.exists():
        print(f"✓ {specimen_path.name}: No data to migrate")
        return True
    
    # Create a run with timestamp based on file modification time
    if old_metrics.exists():
        mtime = old_metrics.stat().st_mtime
        run_id = datetime.fromtimestamp(mtime).strftime("%Y%m%d_%H%M%S")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Move files
    migrated = []
    if old_metrics.exists():
        shutil.copy2(old_metrics, run_dir / "metrics.parquet")
        migrated.append("metrics.parquet")
    
    if old_manifest.exists():
        shutil.copy2(old_manifest, run_dir / "manifest.msgpack")
        migrated.append("manifest.msgpack")
    
    # Copy any other files from strata
    for item in old_strata.iterdir():
        if item.is_file() and item.name not in ["metrics.parquet", "manifest.msgpack"]:
            shutil.copy2(item, run_dir / item.name)
            migrated.append(item.name)
    
    print(f"✓ {specimen_path.name}: Migrated to run {run_id}")
    print(f"  Files: {', '.join(migrated)}")
    
    return True


def main():
    """Main migration function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate specimens from legacy structure to runs structure"
    )
    parser.add_argument(
        "specimen",
        nargs="?",
        type=Path,
        help="Path to specific specimen (or omit to migrate all)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without actually migrating"
    )
    
    args = parser.parse_args()
    
    specimens_dir = Path(__file__).parent.parent / "specimens"
    
    if args.specimen:
        # Migrate specific specimen
        specimen_path = Path(args.specimen)
        if not specimen_path.is_absolute():
            specimen_path = specimens_dir / specimen_path
        
        if args.dry_run:
            print(f"[DRY RUN] Would migrate: {specimen_path}")
        else:
            migrate_specimen(specimen_path)
    else:
        # Migrate all specimens
        if not specimens_dir.exists():
            print(f"✗ Specimens directory not found: {specimens_dir}")
            return
        
        specimens = [d for d in specimens_dir.iterdir() if d.is_dir()]
        
        if args.dry_run:
            print(f"[DRY RUN] Would migrate {len(specimens)} specimens:")
            for spec in specimens:
                old_strata = spec / "strata"
                if old_strata.exists():
                    print(f"  - {spec.name}")
        else:
            print(f"Migrating {len(specimens)} specimens...")
            migrated = 0
            for spec in specimens:
                if migrate_specimen(spec):
                    migrated += 1
            
            print(f"\n✓ Migration complete: {migrated}/{len(specimens)} specimens migrated")


if __name__ == "__main__":
    main()

