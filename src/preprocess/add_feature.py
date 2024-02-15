import polars as pl

from src.base.preprocess.add_feature import BaseFeature
from src.preprocess.initialize import PreprocessInit

class PreprocessAddFeature(BaseFeature, PreprocessInit):
    
    def add_fold_column(self) -> None:
        if not self.inference:
            self.base_data = self.base_data.with_columns(
                pl.col('WEEK_NUM')
                .alias(self.fold_time_col)
                .cast(pl.UInt16)
            )

    def create_feature(self) -> None:
        self.add_fold_column()

    def add_dataset_name_to_feature(self) -> None: 
        self.static_0 = self.static_0.rename(
            {
                col: 'static_0_' + col
                for col in self.static_0.columns
                if col not in self.special_column_list
            }
        )
        self.static_cb_0 = self.static_cb_0.rename(
            {
                col: 'static_cb_0_' + col
                for col in self.static_cb_0.columns
                if col not in self.special_column_list
            }
        )

    def merge_all(self) -> None:
        self.add_dataset_name_to_feature()
        
        self.data = self.base_data
        
        if not self.inference:
            n_rows_begin = self._collect_item_utils(
                self.data.select(pl.count())
            )

        self.data = self.data.join(
            self.static_0, how='left', 
            on=['case_id']
        )
        self.data = self.data.join(
            self.static_cb_0, how='left', 
            on=['case_id']
        )
        
        if not self.inference:
            n_rows_end = self._collect_item_utils(
                self.data.select(pl.count())
            )
            assert n_rows_begin == n_rows_end

