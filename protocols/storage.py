"""●COMPONENT|Ψ:storage_layer|Ω:persist_experiment_data"""

from pathlib import Path
from typing import Optional, Tuple

import msgpack
import numpy as np
import polars as pl
import zarr


class SpecimenStorage:
    """●COMPONENT|Ψ:specimen_persistence|Ω:write_read_experiment_artifacts"""
    
    def __init__(self, specimen_path: Path) -> None:
        """●METHOD|input:Path|output:None|operation:initialize_storage_create_strata"""
        self.specimen_path = Path(specimen_path)
        self.strata_path = self.specimen_path / "strata"
        
        # Auto-create strata directory
        try:
            self.strata_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise RuntimeError(f"Failed to create strata directory at {self.strata_path}: {e}")
    
    def write_manifest(self, metadata: dict) -> None:
        """●METHOD|input:dict|output:None|operation:save_metadata_as_msgpack"""
        manifest_path = self.strata_path / "manifest.msgpack"
        
        try:
            with open(manifest_path, "wb") as f:
                msgpack.pack(metadata, f)
        except Exception as e:
            raise RuntimeError(f"Failed to write manifest to {manifest_path}: {e}")
    
    def write_metrics(self, data: dict) -> None:
        """●METHOD|input:dict|output:None|operation:convert_dict_to_parquet_via_polars"""
        metrics_path = self.strata_path / "metrics.parquet"
        
        try:
            # Convert dict to Polars DataFrame
            df = pl.DataFrame(data)
            # Save as Parquet
            df.write_parquet(metrics_path)
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
    
    def read_metrics(self) -> pl.DataFrame:
        """●METHOD|input:None|output:DataFrame|operation:load_parquet_via_polars"""
        metrics_path = self.strata_path / "metrics.parquet"
        
        if not metrics_path.exists():
            raise FileNotFoundError(f"Metrics file not found at {metrics_path}")
        
        try:
            return pl.read_parquet(metrics_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read metrics from {metrics_path}: {e}")
    
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
