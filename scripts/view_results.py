#!/usr/bin/env python3
"""●COMPONENT|Ψ:universal_results_viewer|Ω:comprehensive_experiment_analysis

Universal Results Viewer for Specimen Vault

Provides comprehensive analysis and visualization of experiment results
across all specimen types. Automatically detects metrics structure and
provides appropriate statistics and insights.

Usage:
    # View specific specimen
    python scripts/view_results.py specimens/2024_12_20_vn_noise_robustness
    
    # View all specimens
    python scripts/view_results.py --all
    
    # Compare specimens
    python scripts/view_results.py --compare specimen1 specimen2
    
    # Export to JSON
    python scripts/view_results.py specimen --json output.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import msgpack
import numpy as np
import polars as pl

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from protocols.storage import SpecimenStorage


# ============================================================================
# SECTION 1: METRIC DETECTION AND CLASSIFICATION
# ============================================================================

def classify_metrics(df: pl.DataFrame) -> Dict[str, Any]:
    """●METHOD|input:DataFrame|output:dict|operation:classify_metrics_by_type
    
    Automatically detect metric types and experiment category.
    
    Returns:
        Dict with metric classification
    """
    columns = df.columns
    
    classification = {
        "experiment_type": "unknown",
        "has_encoding_comparison": False,
        "has_noise_robustness": False,
        "has_temporal_data": False,
        "numeric_columns": [],
        "categorical_columns": [],
        "key_metrics": [],
    }
    
    # Detect experiment type
    if "encoding" in columns and "noise_rate" in columns:
        classification["experiment_type"] = "noise_robustness"
        classification["has_noise_robustness"] = True
        classification["has_encoding_comparison"] = True
    elif "nl_purity" in columns and "vn_purity" in columns:
        classification["experiment_type"] = "vn_comparison"
        classification["has_encoding_comparison"] = True
    elif "unique_to_hall_count" in columns or "energy_diff" in columns:
        classification["experiment_type"] = "hallucination_analysis"
    elif "case_id" in columns and "category" in columns:
        classification["experiment_type"] = "test_case_analysis"
    
    # Classify columns
    for col in columns:
        dtype = df[col].dtype
        if dtype in (pl.Int64, pl.Float64, pl.Float32):
            classification["numeric_columns"].append(col)
        elif dtype == pl.Utf8:
            classification["categorical_columns"].append(col)
    
    # Identify key metrics (common patterns)
    key_patterns = [
        "purity", "retention", "stability", "survival", "loss", "accuracy",
        "features", "overlap", "improvement", "reduction", "energy", "diff"
    ]
    
    for col in classification["numeric_columns"]:
        if any(pattern in col.lower() for pattern in key_patterns):
            classification["key_metrics"].append(col)
    
    return classification


# ============================================================================
# SECTION 2: STATISTICAL ANALYSIS
# ============================================================================

def calculate_statistics(df: pl.DataFrame, metric_col: str) -> Dict[str, float]:
    """●METHOD|input:DataFrame_str|output:dict|operation:calculate_comprehensive_stats"""
    if metric_col not in df.columns:
        return {}
    
    series = df[metric_col]
    
    stats = {
        "count": len(series),
        "mean": float(series.mean()),
        "std": float(series.std()),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
    }
    
    # Calculate percentiles
    for p in [10, 90, 95, 99]:
        stats[f"p{p}"] = float(series.quantile(p / 100))
    
    return stats


def analyze_by_group(
    df: pl.DataFrame,
    group_col: str,
    metric_col: str
) -> Dict[str, Dict[str, float]]:
    """●METHOD|input:DataFrame_str_str|output:dict|operation:calculate_group_statistics"""
    if group_col not in df.columns or metric_col not in df.columns:
        return {}
    
    groups = {}
    for group_value in df[group_col].unique().to_list():
        group_df = df.filter(pl.col(group_col) == group_value)
        groups[str(group_value)] = calculate_statistics(group_df, metric_col)
    
    return groups


# ============================================================================
# SECTION 3: EXPERIMENT-SPECIFIC ANALYSIS
# ============================================================================

def analyze_noise_robustness(df: pl.DataFrame) -> Dict[str, Any]:
    """●METHOD|input:DataFrame|output:dict|operation:analyze_noise_robustness_metrics"""
    if "noise_rate" not in df.columns or "encoding" not in df.columns:
        return {}
    
    analysis = {
        "by_noise_rate": {},
        "by_encoding": {},
        "breaking_points": {},
    }
    
    # Analyze by noise rate
    for noise_rate in sorted(df["noise_rate"].unique().to_list()):
        rate_df = df.filter(pl.col("noise_rate") == noise_rate)
        
        for encoding in ["nl", "vn"]:
            encoding_df = rate_df.filter(pl.col("encoding") == encoding)
            if len(encoding_df) == 0:
                continue
            
            key = f"{encoding}_noise_{noise_rate:.2f}"
            analysis["by_noise_rate"][key] = {
                "semantic_retention": float(encoding_df["semantic_retention"].mean()) if "semantic_retention" in encoding_df.columns else None,
                "purity_retention": float(encoding_df["purity_retention"].mean()) if "purity_retention" in encoding_df.columns else None,
                "critical_feature_survival": float(encoding_df["critical_feature_survival"].mean()) if "critical_feature_survival" in encoding_df.columns else None,
                "count": len(encoding_df),
            }
    
    # Calculate breaking points (where retention drops below 80%)
    for encoding in ["nl", "vn"]:
        encoding_df = df.filter(pl.col("encoding") == encoding)
        if "semantic_retention" not in encoding_df.columns:
            continue
        
        # Group by noise rate and calculate mean retention
        rate_retention = {}
        for noise_rate in sorted(encoding_df["noise_rate"].unique().to_list()):
            rate_df = encoding_df.filter(pl.col("noise_rate") == noise_rate)
            mean_retention = float(rate_df["semantic_retention"].mean())
            rate_retention[noise_rate] = mean_retention
        
        # Find breaking point
        breaking_point = None
        for noise_rate in sorted(rate_retention.keys()):
            if rate_retention[noise_rate] < 0.80:
                breaking_point = noise_rate
                break
        
        analysis["breaking_points"][encoding] = {
            "noise_rate": breaking_point,
            "retention_at_breaking": rate_retention.get(breaking_point) if breaking_point else None,
        }
    
    return analysis


def analyze_vn_comparison(df: pl.DataFrame) -> Dict[str, Any]:
    """●METHOD|input:DataFrame|output:dict|operation:analyze_vn_vs_nl_comparison"""
    if "nl_purity" not in df.columns or "vn_purity" not in df.columns:
        return {}
    
    analysis = {
        "overall": {},
        "by_category": {},
        "by_complexity": {},
        "top_improvements": [],
    }
    
    # Overall statistics
    analysis["overall"] = {
        "nl_purity_mean": float(df["nl_purity"].mean()),
        "vn_purity_mean": float(df["vn_purity"].mean()),
        "purity_improvement_mean": float(df["purity_improvement_pct"].mean()) if "purity_improvement_pct" in df.columns else None,
        "feature_reduction_mean": float(df["feature_reduction_pct"].mean()) if "feature_reduction_pct" in df.columns else None,
    }
    
    # By category
    if "category" in df.columns:
        for category in df["category"].unique().to_list():
            cat_df = df.filter(pl.col("category") == category)
            analysis["by_category"][str(category)] = {
                "count": len(cat_df),
                "nl_purity": float(cat_df["nl_purity"].mean()),
                "vn_purity": float(cat_df["vn_purity"].mean()),
                "improvement": float(cat_df["purity_improvement_pct"].mean()) if "purity_improvement_pct" in cat_df.columns else None,
            }
    
    # By complexity
    if "complexity" in df.columns:
        for complexity in ["simple", "medium", "complex"]:
            comp_df = df.filter(pl.col("complexity") == complexity)
            if len(comp_df) == 0:
                continue
            analysis["by_complexity"][complexity] = {
                "count": len(comp_df),
                "nl_purity": float(comp_df["nl_purity"].mean()),
                "vn_purity": float(comp_df["vn_purity"].mean()),
                "improvement": float(comp_df["purity_improvement_pct"].mean()) if "purity_improvement_pct" in comp_df.columns else None,
            }
    
    # Top improvements
    if "purity_improvement_pct" in df.columns and "case_id" in df.columns:
        top_df = df.sort("purity_improvement_pct", descending=True).head(10)
        analysis["top_improvements"] = [
            {
                "case_id": row["case_id"],
                "improvement": float(row["purity_improvement_pct"]),
                "category": row.get("category", "unknown"),
            }
            for row in top_df.iter_rows(named=True)
        ]
    
    return analysis


# ============================================================================
# SECTION 4: DISPLAY FUNCTIONS
# ============================================================================

def print_header(title: str, width: int = 80) -> None:
    """●METHOD|input:str_int|output:None|operation:print_formatted_header"""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_metric_summary(df: pl.DataFrame, classification: Dict[str, Any]) -> None:
    """●METHOD|input:DataFrame_dict|output:None|operation:print_metric_overview"""
    print_header("METRIC OVERVIEW")
    
    print(f"\nExperiment Type: {classification['experiment_type']}")
    print(f"Total Records: {len(df):,}")
    print(f"Numeric Metrics: {len(classification['numeric_columns'])}")
    print(f"Categorical Columns: {len(classification['categorical_columns'])}")
    
    if classification['key_metrics']:
        print(f"\nKey Metrics Detected:")
        for metric in classification['key_metrics'][:10]:
            print(f"  • {metric}")


def print_statistics(df: pl.DataFrame, classification: Dict[str, Any]) -> None:
    """●METHOD|input:DataFrame_dict|output:None|operation:print_comprehensive_statistics"""
    print_header("STATISTICS")
    
    # Print statistics for key metrics
    for metric in classification['key_metrics'][:10]:
        stats = calculate_statistics(df, metric)
        if not stats:
            continue
        
        print(f"\n{metric.upper()}:")
        print(f"  Count:    {stats['count']:,}")
        print(f"  Mean:     {stats['mean']:.4f}")
        print(f"  Std:      {stats['std']:.4f}")
        print(f"  Median:   {stats['median']:.4f}")
        print(f"  Range:    [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  IQR:      [{stats['q25']:.4f}, {stats['q75']:.4f}]")


def print_experiment_specific(df: pl.DataFrame, classification: Dict[str, Any]) -> None:
    """●METHOD|input:DataFrame_dict|output:None|operation:print_experiment_specific_insights"""
    print_header("EXPERIMENT-SPECIFIC ANALYSIS")
    
    if classification['experiment_type'] == 'noise_robustness':
        analysis = analyze_noise_robustness(df)
        
        if analysis.get('breaking_points'):
            print("\nBreaking Points (Retention < 80%):")
            for encoding, bp_data in analysis['breaking_points'].items():
                if bp_data['noise_rate']:
                    print(f"  {encoding.upper()}: {bp_data['noise_rate']*100:.0f}% noise")
                else:
                    print(f"  {encoding.upper()}: No breaking point detected (< 25% tested)")
        
        if analysis.get('by_noise_rate'):
            print("\nRetention by Noise Rate:")
            print(f"  {'Noise Rate':<12} {'NL Retention':<15} {'VN Retention':<15}")
            print("  " + "-" * 42)
            for noise_rate in sorted(set(float(k.split('_')[-1]) for k in analysis['by_noise_rate'].keys())):
                nl_key = f"nl_noise_{noise_rate:.2f}"
                vn_key = f"vn_noise_{noise_rate:.2f}"
                nl_ret = analysis['by_noise_rate'].get(nl_key, {}).get('semantic_retention')
                vn_ret = analysis['by_noise_rate'].get(vn_key, {}).get('semantic_retention')
                if nl_ret is not None and vn_ret is not None:
                    print(f"  {noise_rate*100:>5.0f}%      {nl_ret:>6.3f}         {vn_ret:>6.3f}")
    
    elif classification['experiment_type'] == 'vn_comparison':
        analysis = analyze_vn_comparison(df)
        
        if analysis.get('overall'):
            print("\nOverall Comparison:")
            overall = analysis['overall']
            print(f"  NL Purity:     {overall.get('nl_purity_mean', 0):.4f}")
            print(f"  VN Purity:     {overall.get('vn_purity_mean', 0):.4f}")
            if overall.get('purity_improvement_mean'):
                print(f"  Improvement:   {overall['purity_improvement_mean']:+.2f}%")
            if overall.get('feature_reduction_mean'):
                print(f"  Feature Reduction: {overall['feature_reduction_mean']:.2f}%")
        
        if analysis.get('by_complexity'):
            print("\nBy Complexity:")
            for complexity in ['simple', 'medium', 'complex']:
                if complexity in analysis['by_complexity']:
                    comp = analysis['by_complexity'][complexity]
                    print(f"  {complexity.capitalize()}:")
                    print(f"    NL: {comp['nl_purity']:.4f} | VN: {comp['vn_purity']:.4f} | "
                          f"Improvement: {comp.get('improvement', 0):+.2f}%")
        
        if analysis.get('top_improvements'):
            print("\nTop 10 Improvements:")
            for i, item in enumerate(analysis['top_improvements'][:10], 1):
                print(f"  {i:2d}. {item['case_id']:<30} {item['improvement']:+.2f}% ({item['category']})")


def print_manifest_info(storage: SpecimenStorage) -> None:
    """●METHOD|input:SpecimenStorage|output:None|operation:print_manifest_metadata"""
    try:
        # Show run information
        runs = storage.list_runs()
        if runs:
            print_header("RUN INFORMATION")
            print(f"\nCurrent Run: {storage.run_id}")
            print(f"Total Runs: {len(runs)}")
            if len(runs) > 1:
                print(f"\nAll Runs:")
                for i, run_id in enumerate(runs[:10], 1):  # Show up to 10
                    print(f"  {i}. {run_id}")
                if len(runs) > 10:
                    print(f"  ... and {len(runs) - 10} more")
        
        # Try to load manifest from current run
        manifest_path = storage.run_path / "manifest.msgpack"
        if not manifest_path.exists():
            # Fallback: try specimen-level manifest
            manifest_path = storage.specimen_path / "manifest.msgpack"
        
        if not manifest_path.exists():
            return
        
        with open(manifest_path, "rb") as f:
            manifest = msgpack.unpack(f)
        
        print_header("EXPERIMENT METADATA")
        
        if "run_id" in manifest:
            print(f"\nRun ID: {manifest['run_id']}")
        if "run_timestamp" in manifest:
            print(f"Run Timestamp: {manifest['run_timestamp']}")
        
        if "specimen_id" in manifest:
            print(f"Specimen ID: {manifest['specimen_id']}")
        
        if "taxonomy" in manifest:
            taxonomy = manifest["taxonomy"]
            print(f"Domain: {taxonomy.get('domain', 'unknown')}")
            print(f"Method: {taxonomy.get('method', 'unknown')}")
        
        if "tags" in manifest:
            print(f"Tags: {', '.join(manifest['tags'])}")
        
        if "created" in manifest:
            print(f"Created: {manifest['created']}")
        
        if "timing" in manifest:
            timing = manifest["timing"]
            if "start" in timing:
                print(f"Start: {timing['start']}")
            if "end" in timing:
                print(f"End: {timing['end']}")
        
        if "summary_statistics" in manifest:
            print("\nSummary Statistics:")
            for key, value in manifest["summary_statistics"].items():
                print(f"  {key}: {value}")
    
    except Exception as e:
        print(f"\n⚠ Could not load manifest: {e}")


# ============================================================================
# SECTION 5: MAIN VIEWER
# ============================================================================

def view_specimen(specimen_path: Path, json_output: Optional[Path] = None) -> Dict[str, Any]:
    """●METHOD|input:Path_Path|output:dict|operation:comprehensive_specimen_analysis"""
    # Check for existing runs first
    runs_path = Path(specimen_path) / "runs"
    if not runs_path.exists():
        print(f"✗ No metrics found. No runs exist in {runs_path}")
        return {}
    
    # List existing runs
    runs = [
        d.name for d in runs_path.iterdir()
        if d.is_dir() and (d / "metrics.parquet").exists()
    ]
    runs = sorted(runs, reverse=True)
    
    if not runs:
        print(f"✗ No metrics found. No runs exist in {runs_path}")
        return {}
    
    # Use the latest run
    latest_run_id = runs[0]
    storage = SpecimenStorage(specimen_path, run_id=latest_run_id)
    
    # Load metrics (from latest run)
    try:
        df = storage.read_metrics()
        print(f"\n📊 Loading metrics from run: {storage.run_id}")
    except FileNotFoundError as e:
        runs = storage.list_runs()
        if runs:
            print(f"✗ No metrics found in current run: {storage.run_id}")
            print(f"  Available runs: {', '.join(runs[:5])}")
            if len(runs) > 5:
                print(f"  ... and {len(runs) - 5} more")
        else:
            print(f"✗ No metrics found. No runs exist in {specimen_path / 'runs'}")
        print(f"  Error: {e}")
        return {}
    except Exception as e:
        print(f"✗ Error loading metrics: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {}
    
    # Classify metrics
    classification = classify_metrics(df)
    
    # Print analysis
    print(f"\n{'='*80}")
    print(f" SPECIMEN: {specimen_path.name}")
    print(f"{'='*80}")
    
    print_manifest_info(storage)
    print_metric_summary(df, classification)
    print_statistics(df, classification)
    print_experiment_specific(df, classification)
    
    # Prepare output data
    output_data = {
        "specimen_id": specimen_path.name,
        "classification": classification,
        "statistics": {
            metric: calculate_statistics(df, metric)
            for metric in classification['key_metrics'][:20]
        },
    }
    
    # Add experiment-specific analysis
    if classification['experiment_type'] == 'noise_robustness':
        output_data['noise_analysis'] = analyze_noise_robustness(df)
    elif classification['experiment_type'] == 'vn_comparison':
        output_data['vn_analysis'] = analyze_vn_comparison(df)
    
    # Export to JSON if requested
    if json_output:
        # If path is relative, save in run folder; if absolute, use as-is
        if json_output.is_absolute():
            output_path = json_output
        else:
            # Save in run folder
            output_path = storage.run_path / json_output.name
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n✓ Exported analysis to {output_path}")
    
    return output_data


def view_all_specimens(specimens_dir: Path = Path("specimens")) -> None:
    """●METHOD|input:Path|output:None|operation:view_all_specimens_summary"""
    print_header("ALL SPECIMENS SUMMARY", width=80)
    
    if not specimens_dir.exists():
        print(f"\n✗ Specimens directory not found: {specimens_dir}")
        return
    
    specimens = [d for d in specimens_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    if not specimens:
        print("\nNo specimens found.")
        return
    
    print(f"\nFound {len(specimens)} specimens:\n")
    
    for specimen_path in sorted(specimens):
        try:
            storage = SpecimenStorage(specimen_path)
            df = storage.read_metrics()
            classification = classify_metrics(df)
            
            print(f"{specimen_path.name}:")
            print(f"  Type: {classification['experiment_type']}")
            print(f"  Records: {len(df):,}")
            if classification['key_metrics']:
                print(f"  Key Metrics: {', '.join(classification['key_metrics'][:5])}")
            print()
        
        except FileNotFoundError:
            print(f"{specimen_path.name}: ⚠ No metrics file found\n")
        except Exception as e:
            print(f"{specimen_path.name}: ⚠ Error loading ({type(e).__name__}: {e})\n")


def compare_specimens(specimen_paths: List[Path]) -> None:
    """●METHOD|input:list_Path|output:None|operation:compare_multiple_specimens"""
    print_header("SPECIMEN COMPARISON", width=80)
    
    if len(specimen_paths) < 2:
        print("Need at least 2 specimens to compare.")
        return
    
    # Load all specimens
    specimens_data = []
    for path in specimen_paths:
        try:
            storage = SpecimenStorage(path)
            df = storage.read_metrics()
            classification = classify_metrics(df)
            specimens_data.append({
                "path": path,
                "name": path.name,
                "df": df,
                "classification": classification,
            })
        except Exception as e:
            print(f"⚠ Could not load {path.name}: {e}")
    
    if len(specimens_data) < 2:
        print("Need at least 2 valid specimens to compare.")
        return
    
    # Compare common metrics
    print("\nComparison:")
    print(f"{'Specimen':<40} {'Type':<20} {'Records':<10}")
    print("-" * 70)
    for spec in specimens_data:
        print(f"{spec['name']:<40} {spec['classification']['experiment_type']:<20} {len(spec['df']):<10,}")


# ============================================================================
# SECTION 6: CLI INTERFACE
# ============================================================================

def main() -> None:
    """●METHOD|input:None|output:None|operation:cli_entry_point"""
    parser = argparse.ArgumentParser(
        description="Universal Results Viewer for Specimen Vault",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View specific specimen
  python scripts/view_results.py specimens/2024_12_20_vn_noise_robustness
  
  # View all specimens
  python scripts/view_results.py --all
  
  # Compare specimens
  python scripts/view_results.py --compare specimen1 specimen2
  
  # Export to JSON (saved in run folder)
  python scripts/view_results.py specimen --json analysis.json
        """
    )
    
    parser.add_argument(
        "specimen",
        nargs="?",
        type=Path,
        help="Path to specimen directory"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="View summary of all specimens"
    )
    
    parser.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        metavar="SPECIMEN",
        help="Compare multiple specimens"
    )
    
    parser.add_argument(
        "--json",
        type=Path,
        metavar="OUTPUT",
        help="Export analysis to JSON file (saved in run folder if relative path)"
    )
    
    args = parser.parse_args()
    
    # Determine action
    if args.all:
        view_all_specimens()
    elif args.compare:
        compare_specimens(args.compare)
    elif args.specimen:
        view_specimen(args.specimen, args.json)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

