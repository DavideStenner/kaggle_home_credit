import polars as pl
from typing import Any, Union, Dict

class PreprocessInit():
    def __init__(self, 
            config_dict: dict[str, Any], target_n_lags: int, 
            embarko_skip: int
        ):
        
        self.target_n_lags: int = target_n_lags
        self.config_dict: dict[str, Any] = config_dict
        self.embarko_skip: int = embarko_skip
        self.n_folds: int = config_dict['N_FOLD']
        self.fold_time_col: str = 'date_order_kfold'
        self.depth_dataset: Dict[str, str] = {
            'depth_0': ["static_0", "static_cb_0"]
        }
        self.inference: bool = False
        self.path_file_pattern: str = '*/train_{pattern_file}*.parquet'
        
        self.anagraphical_column_list: list[str] = [
            'district_544M', 'profession_152M', 'name_4527232M',
            'name_4917606M', 'employername_160M', 
            'contaddr_district_15M', 'contaddr_zipcode_807M', 
            'empladdr_zipcode_114M', 'registaddr_district_1083M',
            'registaddr_zipcode_184M', 'addres_district_368M',
            'addres_zip_823M', 'empls_employer_name_740M'
        ]

        self._initialiaze_empty_dataset()
        
    def _initialiaze_empty_dataset(self):
        self.data: Union[pl.LazyFrame, pl.DataFrame] = None
        self.static_0: Union[pl.LazyFrame, pl.DataFrame] = None
        self.static_cb_0: Union[pl.LazyFrame, pl.DataFrame] = None
    
    def _collect_item_utils(self, data: Union[pl.DataFrame, pl.LazyFrame]) -> Any:
        if isinstance(data, pl.LazyFrame):
            return data.collect().item()
        else:
            return data.item()