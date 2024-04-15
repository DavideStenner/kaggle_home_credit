import polars as pl
from typing import Any, Union, Dict

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
            'static_cb_0_assignmentdate_238D',
            'static_cb_0_assignmentdate_4527235D', 'static_cb_0_responsedate_1012D',
            'static_cb_0_responsedate_4527233D', 'static_cb_0_responsedate_4917613D',
            'tax_registry_a_1_recorddate_4527225D'
        ]
        #calculate day diff and year diff to date_decision
        self.calc_also_year_dates_date_decision: list[str] = [
            'static_0_datefirstoffer_1144D', 'static_0_datelastinstal40dpd_247D',
            'static_0_dtlastpmtallstes_4499206D', 'static_0_firstclxcampaign_1125D',
            'static_0_firstdatedue_489D', 'static_0_lastactivateddate_801D',
            'static_0_lastapplicationdate_877D', 'static_0_lastapprdate_640D',
            'static_0_lastdelinqdate_224D', 'static_0_lastrejectdate_50D',
            'static_0_maxdpdinstldate_3546855D', 'static_cb_0_assignmentdate_238D',
            'static_cb_0_assignmentdate_4955616D', 'static_0_lastrepayingdate_696D',
            'person_1_birth_259D', 'person_1_empl_employedfrom_271D',
        ]
        self.mapper_mask: Dict[str, Dict[str, int]] = None
        self.mapper_dtype: Dict[str, Dict[str, str]] = None
        self.mapper_statistic: Dict[str, Dict[str, float]] = None
        self.used_dataset: list[str] = (
            config_dict['DEPTH_0'] + config_dict['DEPTH_1'] + config_dict['DEPTH_2']
        )
        self.numerical_aggregator: list[pl.Expr] = [
            pl.min, pl.max, pl.mean, pl.std
        ]
        self.date_aggregator: list[pl.Expr] = [
            pl.min, pl.max
        ]
        self._initialize_empty_dataset()
        self._correct_list_date_col()
        
    def _correct_list_date_col(self):
        self.negative_allowed_dates_date_decision = [
            col for col in self.negative_allowed_dates_date_decision
            if any([dataset in col for dataset in self.used_dataset])
        ]
        self.calc_also_year_dates_date_decision = [
            col for col in self.calc_also_year_dates_date_decision
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