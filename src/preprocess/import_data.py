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
        self.static_0: pl.LazyFrame = read_multiple_parquet(
            self.path_file_pattern.format(pattern_file='static_0'),
            root_dir=self.config_dict['PATH_ORIGINAL_DATA'], 
            scan=True
        )
        
        self.static_cb_0: pl.LazyFrame = read_multiple_parquet(
            self.path_file_pattern.format(pattern_file='static_cb_0'),
            root_dir=self.config_dict['PATH_ORIGINAL_DATA'], 
            scan=True
        )
    
    def remap_downcast(self, 
            data: Union[pl.LazyFrame, pl.DataFrame], 
            dataset_name: str
        ) -> Union[pl.LazyFrame, pl.DataFrame]:
        
        #don't remap empty columns as polars raise errors
        column_dataset = (
            set(self.mapper_mask[dataset_name].keys())
            .intersection(
                data.columns
            )
        )
        empty_columns_list = [
            column for column, value_missing in
            (
                data.select(
                    pl.col(column_dataset)
                    .is_null().mean()
                ).collect()
                .to_dicts()[0]
                .items()
            )
            if value_missing == 1.
        ]

        
        data = data.with_columns(
            [
                (
                    (
                        pl.col(col).replace(mapping_dict, default=None)
                    ).cast(pl.UInt64)
                    if col not in empty_columns_list
                    #if it's empty don't apply replace as it raise errors
                    else pl.col(col).cast(pl.UInt64)
                )
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
    
    def downcast_static_0(self):
        #replace categorical before downcasting
        self.static_0 = self.remap_downcast(
            data=self.static_0, dataset_name='static_0'
        )

    def downcast_static_cb_0(self):
        self.static_cb_0 = self.remap_downcast(
            data=self.static_cb_0, dataset_name='static_cb_0'
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
    
    def skip_useless_null_columns(self):
        #drop original useless null columns as name of employer
        self.static_0 = self.static_0.drop(
            [
                col 
                for col in self.mapper_statistic['static_0'].keys()
                if (self.mapper_statistic['static_0'][col]>=self.null_threshold) & (col in self.static_0.columns)
            ]
        )
        self.static_cb_0 = self.static_cb_0.drop(
            [
                col 
                for col in self.mapper_statistic['static_cb_0'].keys() 
                if (self.mapper_statistic['static_cb_0'][col]>=self.null_threshold) & (col in self.static_cb_0.columns)
            ]
        )

    def skip_useless_columns(self):
        #drop useless columns as name of employer
        self.static_0 = self.static_0.drop(
            [col for col in self.static_0.columns if col in self.useless_column_list]
        )
        self.static_cb_0 = self.static_cb_0.drop(
            [col for col in self.static_cb_0.columns if col in self.useless_column_list]
        )
    
    def import_all(self) -> None:
        self.scan_all_dataset()
        self._import_all_mapper()


        self.downcast_base()
        self.downcast_static_0()
        self.downcast_static_cb_0()
        
        self.skip_useless_columns()
        # self.skip_useless_null_columns()