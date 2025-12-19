#!/usr/bin/env python3
"""●COMPONENT|Ψ:vault_indexer|Ω:rebuild_duckdb_catalog_from_specimens"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import msgpack


def scan_manifests(specimens_dir: Path) -> Tuple[List[Dict], List[str]]:
    """●METHOD|input:Path|output:tuple[list_dict_list_str]|operation:load_all_manifests_log_errors"""
    manifests = []
    warnings = []
    
    if not specimens_dir.exists():
        warnings.append(f"Specimens directory not found: {specimens_dir}")
        return manifests, warnings
    
    for specimen_path in specimens_dir.iterdir():
        if not specimen_path.is_dir():
            continue
        
        # Skip hidden directories and .gitkeep files
        if specimen_path.name.startswith('.'):
            continue
        
        manifest_path = specimen_path / "manifest.msgpack"
        
        if not manifest_path.exists():
            warnings.append(f"Missing manifest: {specimen_path.name}")
            continue
        
        try:
            with open(manifest_path, "rb") as f:
                manifest_data = msgpack.unpack(f)
            
            # Validate required fields
            required_fields = ["specimen_id", "created", "taxonomy", "status"]
            missing_fields = [field for field in required_fields if field not in manifest_data]
            
            if missing_fields:
                warnings.append(
                    f"Manifest {specimen_path.name} missing fields: {', '.join(missing_fields)}"
                )
                continue
            
            # Add path to manifest data
            manifest_data["path"] = str(specimen_path.resolve())
            manifests.append(manifest_data)
            
        except Exception as e:
            warnings.append(f"Corrupt manifest {specimen_path.name}: {e}")
            continue
    
    return manifests, warnings


def rebuild_catalog(manifests: List[Dict], vault_path: Path) -> int:
    """●METHOD|input:list_dict_Path|output:int|operation:create_duckdb_table_from_manifests"""
    
    # Remove existing database to ensure clean rebuild
    if vault_path.exists():
        vault_path.unlink()
    
    try:
        conn = duckdb.connect(str(vault_path))
        
        # Create catalog table
        conn.execute("""
            CREATE TABLE catalog (
                specimen_id VARCHAR PRIMARY KEY,
                created TIMESTAMP,
                domain VARCHAR,
                method VARCHAR,
                model VARCHAR,
                tags VARCHAR[],
                status VARCHAR,
                path VARCHAR
            )
        """)
        
        # Insert manifest data
        for manifest in manifests:
            taxonomy = manifest.get("taxonomy", {})
            domain = taxonomy.get("domain", None)
            method = taxonomy.get("method", None)
            model = manifest.get("model", None)
            
            # Convert tags list to array
            tags = manifest.get("tags", [])
            
            conn.execute("""
                INSERT INTO catalog (specimen_id, created, domain, method, model, tags, status, path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                manifest["specimen_id"],
                manifest["created"],
                domain,
                method,
                model,
                tags,
                manifest["status"],
                manifest["path"]
            ])
        
        conn.close()
        return len(manifests)
        
    except Exception as e:
        raise RuntimeError(f"Failed to rebuild catalog: {e}")


def create_metric_views(specimens_dir: Path, vault_path: Path) -> Tuple[int, List[str]]:
    """●METHOD|input:Path_Path|output:tuple[int_list_str]|operation:create_views_for_metrics_parquet"""
    views_created = 0
    warnings = []
    
    try:
        conn = duckdb.connect(str(vault_path))
        
        for specimen_path in specimens_dir.iterdir():
            if not specimen_path.is_dir():
                continue
            
            # Skip hidden directories
            if specimen_path.name.startswith('.'):
                continue
            
            metrics_path = specimen_path / "strata" / "metrics.parquet"
            
            if not metrics_path.exists():
                continue
            
            specimen_id = specimen_path.name
            view_name = f"exp_{specimen_id}"
            
            try:
                # Create view for this specimen's metrics
                conn.execute(
                    f"CREATE OR REPLACE VIEW {view_name} AS "
                    f"SELECT * FROM read_parquet('{metrics_path}')"
                )
                views_created += 1
                
            except Exception as e:
                warnings.append(f"Failed to create view for {specimen_id}: {e}")
                continue
        
        conn.close()
        
    except Exception as e:
        warnings.append(f"Failed to connect to vault database: {e}")
    
    return views_created, warnings


def main() -> None:
    """●METHOD|input:None|output:None|operation:orchestrate_vault_indexing_print_summary"""
    
    specimens_dir = Path("specimens")
    vault_path = Path("vault.duckdb")
    
    print("●PROCESS|operation:indexing_vault|phase:scanning_manifests")
    
    # Scan all manifests
    manifests, scan_warnings = scan_manifests(specimens_dir)
    
    # Print warnings from scanning
    for warning in scan_warnings:
        print(f"⚠ Warning: {warning}", file=sys.stderr)
    
    if not manifests:
        print("✗ No valid specimens found to index", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Found {len(manifests)} valid specimen(s)")
    
    # Rebuild catalog
    print("●PROCESS|operation:indexing_vault|phase:rebuilding_catalog")
    
    try:
        indexed_count = rebuild_catalog(manifests, vault_path)
        print(f"  Indexed {indexed_count} specimen(s) in catalog")
    except Exception as e:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create metric views
    print("●PROCESS|operation:indexing_vault|phase:creating_views")
    
    views_created, view_warnings = create_metric_views(specimens_dir, vault_path)
    
    # Print warnings from view creation
    for warning in view_warnings:
        print(f"⚠ Warning: {warning}", file=sys.stderr)
    
    print(f"  Created {views_created} view(s)")
    
    # Print summary
    print()
    print(f"✓ Indexed {indexed_count} specimens, {views_created} views created")
    print(f"  Vault: {vault_path.resolve()}")


if __name__ == "__main__":
    main()
