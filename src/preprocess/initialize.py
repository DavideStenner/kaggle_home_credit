import polars as pl
from typing import Any, Union, Dict, Callable

from src.base.preprocess.initialize import BaseInit

class PreprocessInit(BaseInit):
    def __init__(self, 
            config_dict: dict[str, Any],
            embarko_skip: int
        ):
        self.config_dict: dict[str, Any] = config_dict
        self.embarko_skip: int = embarko_skip
        self.n_folds: int = config_dict['N_FOLD']
        self.fold_time_col: str = 'date_order_kfold'

        self.inference: bool = False
        self.path_file_pattern: str = '*/train_{pattern_file}*.parquet'
        
        self.special_column_list: list[str] = config_dict['SPECIAL_COLUMNS']
        #hashed value for null in string
        self.hashed_missing_label: str = "a55475b1"
        #for this dates doesn't correct to blank when negative
        self.negative_allowed_dates_date_decision: list[str] = [
            'tax_registry_a_1_recorddate_4527225D',
            'static_cb_0_responsedate_4527233D'
        ]
        self.mapper_mask: Dict[str, Dict[str, int]] = None
        self.mapper_dtype: Dict[str, Dict[str, str]] = None
        self.mapper_statistic: Dict[str, Dict[str, float]] = None
        self.used_dataset: list[str] = (
            config_dict['DEPTH_0'] + config_dict['DEPTH_1'] + config_dict['DEPTH_2']
        )
        self.list_join_expression: list[Union[pl.LazyFrame, pl.DataFrame]] = []
        
        def minnozero(col: str) -> pl.Expr:
            return pl.col(col).filter(pl.col(col)!=0).min()
            
        self.numerical_aggregator: list[Callable[..., pl.Expr]] = [
            minnozero, pl.max, pl.mean, pl.std, pl.sum
        ]
        self._initialize_empty_dataset()
        self._correct_list_date_col()
        self._initialize_filter_expression()
        self._initialize_date_expression()
        
    def _initialize_date_expression(self) -> None:
        def date_min(col) -> pl.Expr:
            return pl.col(col).min()

        def date_max(col) -> pl.Expr:
            return pl.col(col).max()
        
        def date_mean(col) -> pl.Expr:
            return pl.col(col).mean()
        
        self.date_aggregator: list[Callable[..., pl.Expr]] = [
            date_min, date_max, date_mean
        ]

    def _initialize_filter_expression(self) -> None:
        def filtered_min(col: str, pl_filter: pl.Expr) -> pl.Expr:
            return pl.col(col).filter(
                (pl.col(col)!=0) &
                pl_filter
            ).min()
        
        def filtered_max(col: str, pl_filter: pl.Expr) -> pl.Expr:
            return pl.col(col).filter(pl_filter).max()
                
        def filtered_mean(col: str, pl_filter: pl.Expr) -> pl.Expr:
            return pl.col(col).filter(pl_filter).mean()

        def filtered_std(col: str, pl_filter: pl.Expr) -> pl.Expr:
            return pl.col(col).filter(pl_filter).std()

        def filtered_sum(col: str, pl_filter: pl.Expr) -> pl.Expr:
            return pl.col(col).filter(pl_filter).sum()
        
        def filtered_median(col: str, pl_filter: pl.Expr) -> pl.Expr:
            return pl.col(col).filter(pl_filter).median()
                
        self.numerical_filter_aggregator: list[Callable[..., pl.Expr]] = [
            filtered_min, filtered_max, filtered_mean, filtered_std, filtered_sum
        ]
        
    def _correct_list_date_col(self):
        self.negative_allowed_dates_date_decision = [
            col for col in self.negative_allowed_dates_date_decision
            if any([dataset in col for dataset in self.used_dataset])
        ]

        
    def _initialize_empty_dataset(self):
        self.base_data: Union[pl.LazyFrame, pl.DataFrame] = None
        self.data: Union[pl.LazyFrame, pl.DataFrame] = None
        self.static_0: Union[pl.LazyFrame, pl.DataFrame] = None
        self.static_cb_0: Union[pl.LazyFrame, pl.DataFrame] = None
        
        self.person_1: Union[pl.LazyFrame, pl.DataFrame] = None
        self.applprev_1: Union[pl.LazyFrame, pl.DataFrame] = None
        self.other_1: Union[pl.LazyFrame, pl.DataFrame] = None
        self.tax_registry_a_1: Union[pl.LazyFrame, pl.DataFrame] = None
        self.tax_registry_b_1: Union[pl.LazyFrame, pl.DataFrame] = None
        self.tax_registry_c_1: Union[pl.LazyFrame, pl.DataFrame] = None
        self.deposit_1: Union[pl.LazyFrame, pl.DataFrame] = None
        self.debitcard_1: Union[pl.LazyFrame, pl.DataFrame] = None
        self.credit_bureau_a_1: Union[pl.LazyFrame, pl.DataFrame] = None
        self.credit_bureau_b_1: Union[pl.LazyFrame, pl.DataFrame] = None
        
        self.person_2: Union[pl.LazyFrame, pl.DataFrame] = None
        self.applprev_2: Union[pl.LazyFrame, pl.DataFrame] = None
        self.credit_bureau_a_2: Union[pl.LazyFrame, pl.DataFrame] = None
        self.credit_bureau_b_2: Union[pl.LazyFrame, pl.DataFrame] = None
        
    def _collect_item_utils(self, data: Union[pl.DataFrame, pl.LazyFrame]) -> Any:
        if isinstance(data, pl.LazyFrame):
            return data.collect().item()
        else:
            return data.item()