"""●COMPONENT|Ψ:storage_layer|Ω:persist_experiment_data"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import msgpack
import numpy as np
import polars as pl
import zarr


class SpecimenStorage:
    """●COMPONENT|Ψ:specimen_persistence|Ω:write_read_experiment_artifacts
    
    Supports multiple runs per experiment. Each run gets its own timestamped directory
    under runs/ to preserve complete history.
    """
    
    def __init__(self, specimen_path: Path, run_id: Optional[str] = None) -> None:
        """●METHOD|input:Path_Optional[str]|output:None|operation:initialize_storage_create_run_directory
        
        Args:
            specimen_path: Path to specimen directory
            run_id: Optional run ID (timestamp). If None, creates a new run.
        """
        self.specimen_path = Path(specimen_path)
        self.runs_path = self.specimen_path / "runs"
        
        # Create runs directory if it doesn't exist
        try:
            self.runs_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create runs directory at {self.runs_path}: {e}")
        
        # Determine current run directory
        if run_id is None:
            # Create new run with timestamp
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_id = run_id
        self.run_path = self.runs_path / run_id
        
        # Create run directory
        try:
            self.run_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create run directory at {self.run_path}: {e}")
        
        # Backward compatibility: also set strata_path for legacy code
        self.strata_path = self.run_path
    
    def write_manifest(self, metadata: dict, overwrite: bool = False) -> Path:
        """●METHOD|input:dict|output:Path|operation:save_metadata_as_msgpack
        
        Args:
            metadata: Dictionary to save as msgpack
            overwrite: If True, overwrite existing file. If False, raise error (runs never overwrite).
            
        Returns:
            Path to the saved manifest file
        """
        manifest_path = self.run_path / "manifest.msgpack"
        
        # Never overwrite within a run (each run is immutable)
        if manifest_path.exists() and not overwrite:
            raise RuntimeError(
                f"Manifest already exists at {manifest_path}. "
                f"Each run is immutable. Create a new run instead."
            )
        
        # Add run metadata
        metadata = metadata.copy()
        metadata["run_id"] = self.run_id
        metadata["run_timestamp"] = datetime.now().isoformat()
        
        try:
            with open(manifest_path, "wb") as f:
                msgpack.pack(metadata, f)
            return manifest_path
        except Exception as e:
            raise RuntimeError(f"Failed to write manifest to {manifest_path}: {e}")
    
    def write_metrics(self, data: dict, overwrite: bool = False) -> Path:
        """●METHOD|input:dict|output:Path|operation:convert_dict_to_parquet_via_polars
        
        Args:
            data: Dictionary with lists of values (will be converted to DataFrame)
            overwrite: If True, overwrite existing file. If False, raise error (runs never overwrite).
            
        Returns:
            Path to the saved metrics file
        """
        metrics_path = self.run_path / "metrics.parquet"
        
        # Never overwrite within a run (each run is immutable)
        if metrics_path.exists() and not overwrite:
            raise RuntimeError(
                f"Metrics already exist at {metrics_path}. "
                f"Each run is immutable. Create a new run instead."
            )
        
        try:
            # Convert dict to Polars DataFrame
            df = pl.DataFrame(data)
            # Add run_id column to metrics
            df = df.with_columns(pl.lit(self.run_id).alias("run_id"))
            # Save as Parquet
            df.write_parquet(metrics_path)
            return metrics_path
        except Exception as e:
            raise RuntimeError(f"Failed to write metrics to {metrics_path}: {e}")
    
    def write_tensors(
        self, 
        name: str, 
        array: np.ndarray, 
        chunks: Optional[Tuple[int, ...]] = None
    ) -> Path:
        """●METHOD|input:str_ndarray_chunks|output:Path|operation:save_array_as_zarr_chunked"""
        tensor_path = self.strata_path / f"{name}.zarr"
        
        try:
            # Set reasonable default chunks if not provided
            if chunks is None:
                if array.ndim == 2:
                    chunks = (min(1000, array.shape[0]), min(1000, array.shape[1]))
                elif array.ndim == 1:
                    chunks = (min(1000, array.shape[0]),)
                else:
                    # For higher dimensions, chunk along first two dims
                    chunks = tuple(min(1000, s) for s in array.shape[:2]) + array.shape[2:]
            
            # Save as Zarr with chunking using zarr.open_array
            z = zarr.open_array(
                str(tensor_path),
                mode='w',
                shape=array.shape,
                chunks=chunks,
                dtype=array.dtype
            )
            z[:] = array
            
            return tensor_path
        except Exception as e:
            raise RuntimeError(f"Failed to write tensor '{name}' to {tensor_path}: {e}")
    
    def read_metrics(self, run_id: Optional[str] = None) -> pl.DataFrame:
        """●METHOD|input:Optional[str]|output:DataFrame|operation:load_parquet_via_polars
        
        Args:
            run_id: Optional run ID. If None, reads from current run or latest run.
        
        Returns:
            Polars DataFrame with metrics
        """
        if run_id is None:
            # Use current run path
            metrics_path = self.run_path / "metrics.parquet"
        else:
            # Use specified run
            metrics_path = self.runs_path / run_id / "metrics.parquet"
        
        if not metrics_path.exists():
            # Fallback: try to find latest run
            if run_id is None:
                latest_run = self.get_latest_run()
                if latest_run:
                    metrics_path = self.runs_path / latest_run / "metrics.parquet"
                else:
                    raise FileNotFoundError(
                        f"No metrics file found. No runs exist in {self.runs_path}"
                    )
            else:
                raise FileNotFoundError(f"Metrics file not found at {metrics_path}")
        
        try:
            return pl.read_parquet(metrics_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read metrics from {metrics_path}: {e}")
    
    def list_runs(self) -> List[str]:
        """●METHOD|input:None|output:List[str]|operation:list_all_run_ids
        
        Returns:
            List of run IDs (timestamp strings), sorted newest first
        """
        if not self.runs_path.exists():
            return []
        
        runs = [
            d.name for d in self.runs_path.iterdir()
            if d.is_dir() and (d / "metrics.parquet").exists()
        ]
        return sorted(runs, reverse=True)
    
    def get_latest_run(self) -> Optional[str]:
        """●METHOD|input:None|output:Optional[str]|operation:get_most_recent_run_id
        
        Returns:
            Run ID of most recent run, or None if no runs exist
        """
        runs = self.list_runs()
        return runs[0] if runs else None
    
    def read_all_runs(self) -> pl.DataFrame:
        """●METHOD|input:None|output:DataFrame|operation:load_metrics_from_all_runs
        
        Returns:
            Combined DataFrame with metrics from all runs (includes run_id column)
        """
        runs = self.list_runs()
        if not runs:
            raise FileNotFoundError(f"No runs found in {self.runs_path}")
        
        dfs = []
        for run_id in runs:
            run_path = self.runs_path / run_id / "metrics.parquet"
            if run_path.exists():
                df = pl.read_parquet(run_path)
                # Ensure run_id column exists
                if "run_id" not in df.columns:
                    df = df.with_columns(pl.lit(run_id).alias("run_id"))
                dfs.append(df)
        
        if not dfs:
            raise FileNotFoundError(f"No metrics files found in any runs")
        
        return pl.concat(dfs)
    
    def read_tensor_lazy(self, name: str) -> zarr.Array:
        """●METHOD|input:str|output:zarr.Array|operation:return_zarr_handle_without_loading"""
        tensor_path = self.strata_path / f"{name}.zarr"
        
        if not tensor_path.exists():
            raise FileNotFoundError(f"Tensor '{name}' not found at {tensor_path}")
        
        try:
            # Return Zarr array handle (lazy, doesn't load data)
            return zarr.open_array(str(tensor_path), mode='r')
        except Exception as e:
            raise RuntimeError(f"Failed to open tensor '{name}' from {tensor_path}: {e}")
