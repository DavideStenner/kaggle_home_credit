import os
import json
import polars as pl

from typing import Union
from src.base.preprocess.import_data import BaseImport
from src.utils.dtype import TYPE_MAPPING
from src.utils.import_file import read_multiple_parquet
from src.preprocess.initialize import PreprocessInit

class PreprocessImport(BaseImport, PreprocessInit):
    def _import_all_mapper(self):
        self._import_mapper_dtype()
        self._import_mapper_mask()
        self._import_mapper_statistic()
     
    def _import_mapper_dtype(self):
        with open(
            os.path.join(
                self.config_dict['PATH_MAPPER_DATA'],
                'mapper_dtype.json'
            ), 'r'
        ) as file:
            self.mapper_dtype = json.load(file)
    
    def _import_mapper_mask(self):
        with open(
            os.path.join(
                self.config_dict['PATH_MAPPER_DATA'],
                'mapper_mask.json'
            ), 'r'
        ) as file:
            self.mapper_mask = json.load(file)
    
    def _import_mapper_statistic(self):
        with open(
            os.path.join(
                self.config_dict['PATH_MAPPER_DATA'],
                'mapper_statistic.json'
            ), 'r'
        ) as file:
            self.mapper_statistic = json.load(file)

    def scan_all_dataset(self):
        self.base_data: pl.LazyFrame = read_multiple_parquet(
            self.path_file_pattern.format(pattern_file='base'),
            root_dir=self.config_dict['PATH_ORIGINAL_DATA'], 
            scan=True
        )
        filter_credit_bureau_a_2: pl.Expr = (
            #order of top > 0
            (
                (pl.col('pmts_dpd_303P') > 0) |
                (pl.col('pmts_overdue_1152A') > 0) |
                (pl.col('collater_valueofguarantee_876L') > 0) |
                (pl.col('collater_valueofguarantee_1124L') > 0) |
                (pl.col('pmts_overdue_1140A') > 0) |
                (pl.col('pmts_dpd_1073P') > 0)
            ) |
            (
                pl.col('subjectroles_name_541M') != 
                self.hashed_missing_label
            ) |
            (
                pl.col('subjectroles_name_838M') != 
                self.hashed_missing_label          
            )
        )
        for dataset in  self.used_dataset:
            filter_pl = (
                filter_credit_bureau_a_2 if dataset == 'credit_bureau_a_2'
                else None
            )
            setattr(
                self, 
                dataset, 
                read_multiple_parquet(
                    self.path_file_pattern.format(pattern_file=dataset),
                    root_dir=self.config_dict['PATH_ORIGINAL_DATA'], 
                    scan=True, filter_pl=filter_pl
                )
            )
        
    def remap_downcast(self, 
            data: Union[pl.LazyFrame, pl.DataFrame], 
            dataset_name: str
        ) -> Union[pl.LazyFrame, pl.DataFrame]:
        #don't remap empty columns as polars raise errors
        remap_categories = len(self.mapper_mask[dataset_name].keys())>0
        
        if remap_categories:
            if self.inference:
                skip_empty_dataset = (
                    self._collect_item_utils(
                        data.select(pl.len())
                    ) == 0
                )
                count_null = data.select(
                    pl.col(self.mapper_mask[dataset_name].keys())
                    .is_null().mean()
                )
                if isinstance(data, pl.LazyFrame):
                    count_null = count_null.collect()
                    
                    
                #don't remap empty columns as polars raise errors
                empty_columns_list = [
                    column for column, value_missing in
                    (
                        count_null
                        .to_dicts()[0]
                        .items()
                    )
                    if value_missing == 1.
                ]
            else:
                #during training no problem
                empty_columns_list = []
                skip_empty_dataset = False

            data = data.with_columns(
            [
                (
                    #if it's empty don't apply replace as it raise errors
                    pl.col(col).cast(pl.UInt64)
                    if (col in empty_columns_list) or (skip_empty_dataset)

                    else pl.col(col).replace(mapping_dict, default=None)
                ).cast(pl.UInt64)
                for col, mapping_dict in self.mapper_mask[dataset_name].items()
                if col in data.columns
            ]
        )
        #downcast
        mapper_column_cast = {
            col: TYPE_MAPPING[dtype_str]
            for col, dtype_str in self.mapper_dtype[dataset_name].items()
        }
        data = data.with_columns(
            [
                pl.col(col).cast(mapper_column_cast[col])
                for col in data.columns
            ]
        )
        return data
    
    def downcast_feature_dataset(self):
        for dataset in self.used_dataset:
            setattr(
                self, 
                dataset,  
                self.remap_downcast(
                    data=getattr(self, dataset), dataset_name=dataset
                )
            )
        
    def downcast_base(self):
        self.base_data = self.base_data.with_columns(
            pl.col('case_id').cast(pl.Int32),
            pl.col('date_decision').cast(pl.Date),
            pl.col('MONTH').cast(pl.Int32),
            pl.col('WEEK_NUM').cast(pl.Int16)
        )
        if not self.inference:
            self.base_data = self.base_data.with_columns(
                pl.col('target').cast(pl.UInt8)
            ).filter(
                pl.col('target').is_not_null()
            ).sort(['case_id', 'date_decision'])
    
    def import_all(self) -> None:
        self.scan_all_dataset()
        self._import_all_mapper()


        self.downcast_base()
        self.downcast_feature_dataset()