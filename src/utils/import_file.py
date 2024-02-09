import os

import polars as pl

from glob import glob
from typing import Union

def read_multiple_parquet(
        path_pattern: str, root_dir: str, 
        scan: bool=True
    ) -> Union[None, pl.LazyFrame, pl.DataFrame]:
     
    path_file_list = glob(pathname=path_pattern, root_dir=root_dir)
    num_files = len(path_file_list)
            
    if num_files==0:
        print(f'No dataset founded as {path_pattern}')
        return None
    
    elif num_files == 1:
        path_ = os.path.join(
            root_dir, path_file_list[0]
        )
        data = (
            pl.scan_parquet(path_) if scan
            else pl.read_parquet(path_)
        )
            
    else:

        multiple_scan_list = [
            (
                pl.scan_parquet(
                    os.path.join(
                        root_dir, path_file
                    )
                ) if scan
                else pl.read_parquet(
                    os.path.join(
                        root_dir, path_file
                    )
                )
            )
            for path_file in path_file_list
        ]
        data = pl.concat(multiple_scan_list, how='vertical_relaxed')
        
    return data
