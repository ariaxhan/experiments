#!/usr/bin/env python3
"""●COMPONENT|Ψ:results_analyzer|Ω:query_and_analyze_hallucination_biopsy_data

Simple script to query and analyze results from the hallucination biopsy experiment.
Run this after executing protocol.py to see the results and basic statistics.

Usage:
    python analyze_results.py
"""

from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from protocols.storage import SpecimenStorage
from protocols.query import VaultQuery


def analyze_local_results():
    """●METHOD|input:None|output:None|operation:load_and_display_local_parquet_data
    
    Analyze results directly from the specimen's strata directory.
    This works even before indexing the vault.
    """
    print("=" * 80)
    print("HALLUCINATION BIOPSY RESULTS - LOCAL ANALYSIS")
    print("=" * 80)
    
    # Load specimen storage
    specimen_path = Path(__file__).parent
    storage = SpecimenStorage(specimen_path)
    
    # Check if results exist
    metrics_path = specimen_path / "strata" / "metrics.parquet"
    if not metrics_path.exists():
        print("\n❌ No results found. Run protocol.py first!")
        print(f"   Expected: {metrics_path}")
        return
    
    # Load metrics
    print("\n📊 Loading metrics from Parquet...")
    metrics = storage.read_metrics()
    print(f"   ✓ Loaded {len(metrics)} experiments")
    
    # Display full results table
    print("\n" + "=" * 80)
    print("FULL RESULTS TABLE")
    print("=" * 80)
    print(metrics)
    
    # Basic statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print(f"\n📈 Total Experiments: {len(metrics)}")
    print(f"   Experiment Names: {', '.join(metrics['experiment_name'].to_list())}")
    
    # Feature activation statistics
    print("\n🔬 Feature Activation Counts:")
    print(f"   Fact (avg):          {metrics['fact_total_active'].mean():.1f} features")
    print(f"   Hallucination (avg): {metrics['hall_total_active'].mean():.1f} features")
    print(f"   Range (fact):        {metrics['fact_total_active'].min()}-{metrics['fact_total_active'].max()}")
    print(f"   Range (hall):        {metrics['hall_total_active'].min()}-{metrics['hall_total_active'].max()}")
    
    # Energy statistics
    print("\n⚡ Energy Metrics:")
    print(f"   Fact energy (avg):   {metrics['fact_energy'].mean():.3f}")
    print(f"   Hall energy (avg):   {metrics['hall_energy'].mean():.3f}")
    print(f"   Energy diff (avg):   {metrics['energy_diff'].mean():.3f}")
    print(f"   Energy diff (range): {metrics['energy_diff'].min():.3f} to {metrics['energy_diff'].max():.3f}")
    
    # Biomarker statistics
    print("\n🧬 Hallucination Biomarkers:")
    print(f"   Unique features (avg):  {metrics['unique_to_hall_count'].mean():.1f}")
    print(f"   Unique features (max):  {metrics['unique_to_hall_count'].max()}")
    print(f"   Unique features (min):  {metrics['unique_to_hall_count'].min()}")
    print(f"   Missing features (avg): {metrics['missing_from_hall_count'].mean():.1f}")
    
    # Experiment rankings
    print("\n" + "=" * 80)
    print("EXPERIMENT RANKINGS")
    print("=" * 80)
    
    print("\n🏆 Most Unique Hallucination Features:")
    sorted_by_unique = metrics.sort("unique_to_hall_count", descending=True)
    for i, row in enumerate(sorted_by_unique.iter_rows(named=True), 1):
        print(f"   {i}. {row['experiment_name']}: {row['unique_to_hall_count']} unique features")
    
    print("\n⚡ Largest Energy Differences:")
    sorted_by_energy = metrics.sort("energy_diff", descending=True)
    for i, row in enumerate(sorted_by_energy.iter_rows(named=True), 1):
        sign = "+" if row['energy_diff'] > 0 else ""
        print(f"   {i}. {row['experiment_name']}: {sign}{row['energy_diff']:.3f}")
    
    # Top biomarker features
    print("\n" + "=" * 80)
    print("TOP BIOMARKER FEATURES BY EXPERIMENT")
    print("=" * 80)
    
    for row in metrics.iter_rows(named=True):
        print(f"\n🔬 {row['experiment_name']}:")
        print(f"   Fact: '{row['fact_text']}'")
        print(f"   Hall: '{row['hallucination_text']}'")
        print(f"   Top 3 Biomarker Features:")
        for i in range(1, 4):
            feat_id = row[f'top_feature_{i}']
            feat_words = row[f'top_feature_{i}_words']
            if feat_id != -1:
                print(f"      #{feat_id}: {feat_words}")
    
    # Tensor information
    print("\n" + "=" * 80)
    print("TENSOR DATA")
    print("=" * 80)
    
    fact_acts = storage.read_tensor_lazy("fact_activations")
    hall_acts = storage.read_tensor_lazy("hall_activations")
    
    print(f"\n📦 Fact Activations:")
    print(f"   Shape: {fact_acts.shape}")
    print(f"   Dtype: {fact_acts.dtype}")
    print(f"   Chunks: {fact_acts.chunks}")
    print(f"   Size: ~{fact_acts.nbytes / 1024 / 1024:.1f} MB")
    
    print(f"\n📦 Hallucination Activations:")
    print(f"   Shape: {hall_acts.shape}")
    print(f"   Dtype: {hall_acts.dtype}")
    print(f"   Chunks: {hall_acts.chunks}")
    print(f"   Size: ~{hall_acts.nbytes / 1024 / 1024:.1f} MB")
    
    # Sparsity analysis
    print("\n🔍 Sparsity Analysis:")
    fact_data = fact_acts[:]
    hall_data = hall_acts[:]
    
    fact_nonzero = (fact_data > 0).sum()
    hall_nonzero = (hall_data > 0).sum()
    total_elements = fact_data.size
    
    print(f"   Fact sparsity:  {100 * (1 - fact_nonzero/total_elements):.2f}% zeros")
    print(f"   Hall sparsity:  {100 * (1 - hall_nonzero/total_elements):.2f}% zeros")
    print(f"   Fact active:    {fact_nonzero}/{total_elements} elements")
    print(f"   Hall active:    {hall_nonzero}/{total_elements} elements")


def analyze_vault_results():
    """●METHOD|input:None|output:None|operation:query_indexed_vault_via_sql
    
    Analyze results via SQL queries on the indexed vault.
    This requires running index_vault.py first.
    """
    print("\n" + "=" * 80)
    print("VAULT SQL ANALYSIS")
    print("=" * 80)
    
    try:
        vault = VaultQuery()
        
        # Check if specimen is indexed
        print("\n🔍 Checking vault index...")
        catalog = vault.search("""
            SELECT specimen_id, created, domain, method 
            FROM catalog 
            WHERE specimen_id = '2024_12_19_hallucination_biopsy_gemma2'
        """)
        
        if len(catalog) == 0:
            print("   ⚠️  Specimen not indexed yet. Run: python scripts/index_vault.py")
            return
        
        print(f"   ✓ Found specimen in catalog")
        print(f"   Created: {catalog['created'][0]}")
        print(f"   Domain: {catalog['domain'][0]}")
        print(f"   Method: {catalog['method'][0]}")
        
        # Query experiment results
        print("\n📊 Querying experiment results...")
        results = vault.search("""
            SELECT 
                experiment_name,
                unique_to_hall_count,
                energy_diff,
                top_feature_1,
                top_feature_1_words
            FROM exp_2024_12_19_hallucination_biopsy_gemma2
            ORDER BY unique_to_hall_count DESC
        """)
        
        print("\n🏆 Top Experiments by Unique Features:")
        print(results)
        
        # Aggregate statistics
        print("\n📈 Aggregate Statistics:")
        stats = vault.search("""
            SELECT 
                COUNT(*) as total_experiments,
                AVG(unique_to_hall_count) as avg_unique_features,
                MAX(unique_to_hall_count) as max_unique_features,
                AVG(energy_diff) as avg_energy_diff,
                AVG(fact_total_active) as avg_fact_features,
                AVG(hall_total_active) as avg_hall_features
            FROM exp_2024_12_19_hallucination_biopsy_gemma2
        """)
        
        print(stats)
        
        # Find experiments with high energy differences
        print("\n⚡ High Energy Difference Experiments:")
        high_energy = vault.search("""
            SELECT 
                experiment_name,
                energy_diff,
                fact_energy,
                hall_energy
            FROM exp_2024_12_19_hallucination_biopsy_gemma2
            WHERE ABS(energy_diff) > 0.1
            ORDER BY ABS(energy_diff) DESC
        """)
        
        if len(high_energy) > 0:
            print(high_energy)
        else:
            print("   No experiments with |energy_diff| > 0.1")
        
    except Exception as e:
        print(f"   ❌ Error querying vault: {e}")
        print("   Make sure vault.duckdb exists and is properly indexed.")


def main():
    """●METHOD|input:None|output:None|operation:run_both_analysis_modes"""
    
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "HALLUCINATION BIOPSY ANALYZER" + " " * 29 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Always run local analysis
    analyze_local_results()
    
    # Try vault analysis if available
    try:
        analyze_vault_results()
    except Exception as e:
        print(f"\n⚠️  Vault analysis skipped: {e}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\n💡 Next Steps:")
    print("   1. Index vault: cd ../.. && python scripts/index_vault.py")
    print("   2. Run custom queries using VaultQuery")
    print("   3. Visualize activation differences")
    print("   4. Compare with other specimens")
    print()


if __name__ == "__main__":
    main()

