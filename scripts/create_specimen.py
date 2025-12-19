#!/usr/bin/env python3
"""●COMPONENT|Ψ:specimen_scaffolder|Ω:create_new_experiment_directory_structure"""

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import msgpack


def validate_specimen_id(specimen_id: str) -> bool:
    """●METHOD|input:str|output:bool|operation:check_format_YYYY_MM_DD_descriptor"""
    # Pattern: YYYY_MM_DD_descriptor (alphanumeric + underscores only)
    pattern = r"^\d{4}_\d{2}_\d{2}_[a-zA-Z0-9_]+$"
    return bool(re.match(pattern, specimen_id))


def check_duplicate(specimen_id: str, specimens_dir: Path) -> bool:
    """●METHOD|input:str_Path|output:bool|operation:verify_specimen_id_not_exists"""
    specimen_path = specimens_dir / specimen_id
    return specimen_path.exists()


def create_specimen_structure(
    specimen_id: str,
    domain: str,
    method: str,
    tags: Optional[List[str]] = None,
    specimens_dir: Path = Path("specimens")
) -> Path:
    """●METHOD|input:str_str_str_tags_Path|output:Path|operation:scaffold_complete_specimen_directory"""
    
    specimen_path = specimens_dir / specimen_id
    strata_path = specimen_path / "strata"
    
    try:
        # Create directory structure
        specimen_path.mkdir(parents=True, exist_ok=False)
        strata_path.mkdir(parents=True, exist_ok=True)
        
        # Create manifest.msgpack
        manifest_data = {
            "specimen_id": specimen_id,
            "created": datetime.now(timezone.utc).isoformat(),
            "taxonomy": {
                "domain": domain,
                "method": method
            },
            "status": "active",
            "artifacts": {},
            "tags": tags if tags else []
        }
        
        manifest_path = specimen_path / "manifest.msgpack"
        with open(manifest_path, "wb") as f:
            msgpack.pack(manifest_data, f)
        
        # Create protocol.py template
        protocol_content = f'''"""●COMPONENT|Ψ:experiment_protocol|Ω:execute_{specimen_id}_experiment"""

from pathlib import Path
from protocols.storage import SpecimenStorage


def run_experiment() -> None:
    """●METHOD|input:None|output:None|operation:execute_experiment_save_results"""
    # Initialize storage for this specimen
    storage = SpecimenStorage(Path(__file__).parent)
    
    # TODO: Implement experiment logic here
    # Example:
    # data = {{"step": [1, 2, 3], "loss": [0.5, 0.3, 0.1]}}
    # storage.write_metrics(data)
    
    pass


if __name__ == "__main__":
    run_experiment()
'''
        
        protocol_path = specimen_path / "protocol.py"
        with open(protocol_path, "w") as f:
            f.write(protocol_content)
        
        # Create field_notes.md template
        field_notes_content = f'''●LOG|specimen:{specimen_id}|created:{manifest_data["created"]}

# Field Notes: {specimen_id}

●SECTION|purpose
Document observations, insights, and methodology for this experiment.

●SECTION|methodology
- Domain: {domain}
- Method: {method}
- Tags: {", ".join(tags) if tags else "none"}

●SECTION|observations
<!-- Record experimental observations here -->

●SECTION|results
<!-- Summarize key findings here -->

●SECTION|next_steps
<!-- Note follow-up experiments or improvements -->
'''
        
        field_notes_path = specimen_path / "field_notes.md"
        with open(field_notes_path, "w") as f:
            f.write(field_notes_content)
        
        return specimen_path
        
    except Exception as e:
        # Clean up on failure
        if specimen_path.exists():
            import shutil
            shutil.rmtree(specimen_path)
        raise RuntimeError(f"Failed to create specimen structure: {e}")


def main() -> None:
    """●METHOD|input:None|output:None|operation:parse_args_create_specimen_print_result"""
    
    parser = argparse.ArgumentParser(
        description="●TOOL|Ψ:specimen_creator|Ω:scaffold_new_experiment_directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
●EXAMPLES:
  %(prog)s --id 2024_01_15_sae_probe --domain interpretability --method sae_analysis
  %(prog)s --id 2024_01_15_gpt2_finetune --domain training --method fine_tuning --tags gpt2,small_scale
        """
    )
    
    parser.add_argument(
        "--id",
        required=True,
        metavar="SPECIMEN_ID",
        help="●PARAM|format:YYYY_MM_DD_descriptor|constraint:alphanumeric_underscores_only"
    )
    
    parser.add_argument(
        "--domain",
        required=True,
        choices=["interpretability", "training", "benchmarking", "alignment"],
        help="●PARAM|taxonomy:domain|options:interpretability|training|benchmarking|alignment"
    )
    
    parser.add_argument(
        "--method",
        required=True,
        choices=["sae_analysis", "probe_training", "ablation_study", "fine_tuning"],
        help="●PARAM|taxonomy:method|options:sae_analysis|probe_training|ablation_study|fine_tuning"
    )
    
    parser.add_argument(
        "--tags",
        required=False,
        metavar="TAG1,TAG2,...",
        help="●PARAM|format:comma_separated|purpose:searchability"
    )
    
    args = parser.parse_args()
    
    # Validate specimen_id format
    if not validate_specimen_id(args.id):
        print(f"✗ Error: Invalid specimen_id format: '{args.id}'", file=sys.stderr)
        print(f"  Expected format: YYYY_MM_DD_descriptor (alphanumeric + underscores only)", file=sys.stderr)
        print(f"  Example: 2024_01_15_sae_probe", file=sys.stderr)
        sys.exit(1)
    
    # Check for duplicates
    specimens_dir = Path("specimens")
    if check_duplicate(args.id, specimens_dir):
        print(f"✗ Error: Specimen '{args.id}' already exists in {specimens_dir.resolve()}", file=sys.stderr)
        sys.exit(1)
    
    # Parse tags
    tags = [tag.strip() for tag in args.tags.split(",")] if args.tags else []
    
    # Create specimen structure
    try:
        specimen_path = create_specimen_structure(
            specimen_id=args.id,
            domain=args.domain,
            method=args.method,
            tags=tags,
            specimens_dir=specimens_dir
        )
        
        print(f"✓ Created specimen: {args.id}")
        print(f"  Path: {specimen_path.resolve()}")
        print(f"  Domain: {args.domain}")
        print(f"  Method: {args.method}")
        if tags:
            print(f"  Tags: {', '.join(tags)}")
        
    except Exception as e:
        print(f"✗ Error: Failed to create specimen: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
