"""●COMPONENT|Ψ:query_layer|Ω:search_retrieve_experiment_data"""

from pathlib import Path
from typing import Optional

import duckdb
import polars as pl


class VaultQuery:
    """●COMPONENT|Ψ:vault_query_engine|Ω:sql_search_across_specimens"""
    
    def __init__(self, vault_path: Path = Path("vault.duckdb")) -> None:
        """●METHOD|input:Path|output:None|operation:connect_duckdb_build_catalog"""
        self.vault_path = Path(vault_path)
        
        try:
            # Connect to DuckDB (creates file if doesn't exist)
            self.conn = duckdb.connect(str(self.vault_path))
            
            # Build catalog of specimens
            self._build_catalog()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize VaultQuery with {self.vault_path}: {e}")
    
    def _build_catalog(self) -> None:
        """●METHOD|input:None|output:None|operation:scan_specimens_create_duckdb_views"""
        specimens_dir = Path("specimens")
        
        # Handle empty vault gracefully
        if not specimens_dir.exists():
            return
        
        try:
            # Scan for all specimen directories containing metrics.parquet
            for specimen_path in specimens_dir.iterdir():
                if not specimen_path.is_dir():
                    continue
                
                metrics_path = specimen_path / "strata" / "metrics.parquet"
                if not metrics_path.exists():
                    continue
                
                # Create view named exp_{specimen_id}
                specimen_id = specimen_path.name
                view_name = f"exp_{specimen_id}"
                
                try:
                    # Create or replace view for this specimen's metrics
                    self.conn.execute(
                        f"CREATE OR REPLACE VIEW {view_name} AS "
                        f"SELECT * FROM read_parquet('{metrics_path}')"
                    )
                except Exception as e:
                    # Log but continue if individual specimen fails
                    print(f"Warning: Failed to create view for specimen {specimen_id}: {e}")
                    continue
        except Exception as e:
            # Handle case where specimens directory is not accessible
            print(f"Warning: Failed to build catalog: {e}")
    
    def search(self, sql: str) -> pl.DataFrame:
        """●METHOD|input:str|output:DataFrame|operation:execute_sql_return_polars_df"""
        try:
            # Execute SQL query and get results
            result = self.conn.execute(sql).fetchall()
            
            # Get column names
            if result:
                columns = [desc[0] for desc in self.conn.description]
                # Create Polars DataFrame from results
                return pl.DataFrame(result, schema=columns, orient="row")
            else:
                # Return empty DataFrame with proper columns
                columns = [desc[0] for desc in self.conn.description]
                return pl.DataFrame(schema=columns)
        except Exception as e:
            raise RuntimeError(f"Failed to execute SQL query: {e}")
    
    def find_by_tag(self, tag: str) -> pl.DataFrame:
        """●METHOD|input:str|output:DataFrame|operation:query_catalog_for_specimens_with_tag"""
        try:
            # Get list of all views (specimens)
            views_result = self.conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='main' AND table_type='VIEW' "
                "AND table_name LIKE 'exp_%'"
            ).fetchall()
            
            if not views_result:
                # Return empty DataFrame if no specimens
                return pl.DataFrame()
            
            # Build UNION query across all specimens
            # Assume each specimen has a 'tags' column or we check for tag in metadata
            union_parts = []
            for (view_name,) in views_result:
                # Extract specimen_id from view name
                specimen_id = view_name.replace("exp_", "")
                union_parts.append(
                    f"SELECT '{specimen_id}' as specimen_id, * FROM {view_name} "
                    f"WHERE tags LIKE '%{tag}%'"
                )
            
            if not union_parts:
                return pl.DataFrame()
            
            # Execute union query
            query = " UNION ALL ".join(union_parts)
            result = self.conn.execute(query).fetchall()
            
            if result:
                columns = [desc[0] for desc in self.conn.description]
                return pl.DataFrame(result, schema=columns, orient="row")
            else:
                return pl.DataFrame()
        except Exception as e:
            # If error (e.g., no 'tags' column), return empty DataFrame gracefully
            print(f"Warning: Failed to find specimens by tag '{tag}': {e}")
            return pl.DataFrame()
    
    def __del__(self) -> None:
        """●METHOD|input:None|output:None|operation:cleanup_close_connection"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
        except Exception:
            pass
