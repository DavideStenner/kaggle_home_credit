import os
import json
import warnings

import polars as pl
import polars.selectors as cs

from typing import Mapping, Union, Tuple, Dict
from tqdm import tqdm

from src.utils.import_file import read_multiple_parquet

TYPE_MAPPING: Mapping[str, pl.DataType] = {
    "float64": pl.Float64,
    "int64": pl.Int64,
    "uint64": pl.UInt64,
    
    "float32": pl.Float32,
    "int32": pl.Int32,
    "uint32": pl.UInt32,

    "int16": pl.Int16,
    "uint16": pl.UInt16,
    
    "int8": pl.Int8,
    "uint8": pl.UInt8,
    
    "str": pl.String,
    "bool": pl.Boolean,
    "date": pl.Date,
}
TYPE_MAPPING_REVERSE = {
    pl_dtype_: type_
    for type_, pl_dtype_ in TYPE_MAPPING.items()
}


def get_mapper_categorical(
        data: Union[pl.LazyFrame, pl.DataFrame], 
        check_cat_length: int = 200,
        message_error: str = 'Categories {col} has over {num_different_cat_values} different values.'
    ) -> Mapping[str, int]:
    """
    check dataset and return int remapped dataset.
    It save the remapper

    Args:
        data (Union[pl.LazyFrame, pl.DataFrame]): dataset
        check_cat_length (int): number of different categories which raise a warning

    Returns:
        Mapping[str, int]: mapping dictionary
    """
    mapper_mask_col = {}
    lazy_mode = isinstance(data, pl.LazyFrame)

    categorical_col = data.select(cs.by_dtype(pl.String)).columns
    for col in  [x for x in categorical_col if x[-1]!='D']:
        
        unique_values = (
            data.select(col).drop_nulls().collect()[col].unique() 
            if lazy_mode 
            else data[col].drop_nulls().unique()
        )
        
        mapper_mask_col[col] = {
            value: i 
            for i, value in enumerate(unique_values.sort().to_list())
        }
        num_different_cat_values = len(mapper_mask_col[col].values())
        if  (num_different_cat_values < 2) | (num_different_cat_values > check_cat_length):
            print(
                message_error.format(
                    col=col, 
                    num_different_cat_values=num_different_cat_values
                )
            )
    
    return mapper_mask_col
    
def get_mapper_numerical(
        data: Union[pl.LazyFrame, pl.DataFrame], categorical_columns: list[str],
        type_mapping_reverse: Mapping[pl.DataType, str]=TYPE_MAPPING_REVERSE,
        threshold: float = 0.001, 
        dtype_choices: Tuple[pl.DataType]=[
            pl.UInt8, pl.Int8, pl.UInt16,  pl.Int16,  pl.UInt32,  pl.Int32, pl.Float32, 
            pl.UInt64, pl.Int64, pl.Float64
        ],
    ) -> Mapping[str, str]:
    """
    Calculate best type on numerical column. Must be executed after get_mapper_categorical

    Args:
        data (Union[pl.LazyFrame, pl.DataFrame]): dataset
        threshold (float, optional): don't allow any float to lose precision more than this value. Any values are ok for GBT algorithms as long as you don't unique values.
            See https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_decimal_values_in_[0,_1]. Defaults to 0.001.
        dtype_choices (Tuple[pl.DataType], optional): list of downcast type. Defaults to [ pl.UInt8, pl.Int8, pl.UInt16,  pl.Int16,  pl.UInt32,  pl.Int32, pl.Float32 ].

    Returns:
        Mapping[str, str]: mapping dtype
    """
    lazy_mode = isinstance(data, pl.LazyFrame)
    
    col_to_check = data.select(cs.by_dtype(pl.NUMERIC_DTYPES)).columns
    mapper_column = {}
    
    for col in tqdm(col_to_check):
        dtype_choices_col = (
            [pl.UInt8, pl.UInt16,  pl.UInt32, pl.UInt64]
            if col in categorical_columns
            else dtype_choices
        )           
            
        check_passed = False
        
        for dtype_ in dtype_choices_col:
            if check_passed:
                continue
            
            try:
                check_passed = data.select(
                    pl.col(col),
                    pl.col(col).cast(dtype=dtype_).alias('doncasted_')
                ).select(
                    pl.all_horizontal(
                        ((pl.col(col)-pl.col('doncasted_')).abs().max() <= threshold) &
                        ((pl.col(col).null_count()-pl.col('doncasted_').null_count()).abs() == 0) &
                        ((pl.col(col).n_unique()-pl.col('doncasted_').n_unique()).mean() <= threshold)
                    )
                )
                if lazy_mode:
                    check_passed = check_passed.collect()
                
                check_passed = check_passed.item()
                    
                if check_passed:
                    mapper_column[col] = type_mapping_reverse[dtype_]
            except:
                check_passed = False
        
        if not check_passed:
            mapper_column[col] = type_mapping_reverse[data.select(col).dtypes[0]]
    
    #add boolean
    for col in  data.select(cs.by_dtype(pl.Boolean)).columns:
        mapper_column[col] = type_mapping_reverse[pl.Boolean]

    #add date
    for col in [x for x in data.columns if x[-1]=='D']:
        mapper_column[col] =type_mapping_reverse[pl.Date]

    return mapper_column


def get_mapper_statistic(
        data: Union[pl.LazyFrame, pl.DataFrame] 
    ) -> Mapping[str, int]:

    mapper_statistic = {}
    lazy_mode = isinstance(data, pl.LazyFrame)

    for col in data.columns:
        
        null_values = (
            (
                data.select(col).null_count().collect().item()/
                data.select(pl.count()).collect().item()
            )
            if lazy_mode
            else data.select(col).null_count().item()/data.shape[0]
        )
        
        mapper_statistic[col] = null_values    
    return mapper_statistic

def get_mapping_info(
        config: Dict[str, str],
        file_name_list: list[str]=[
            'static_0', 'static_cb_0',
            'applprev_1', 'other_1', 'tax_registry_a_1',
            'tax_registry_b_1', 'tax_registry_c_1', 'credit_bureau_a_1',
            'credit_bureau_b_1', 'deposit_1', 'person_1', 'debitcard_1',
            'applprev_2', 'person_2', 'credit_bureau_a_2', 'credit_bureau_b_2',
        ]
    ) -> None:
    
    mapper_col_all_file = {}
    mapper_mask_all_file = {}
    mapper_statistic_all_file = {}
    
    for file_name in file_name_list:
        print(f'\nStarting {file_name}\n')
        data = read_multiple_parquet(
            f'*/*_{file_name}*.parquet',
            root_dir=config['PATH_ORIGINAL_DATA'], 
            scan=False
        )
        size_begin = data.estimated_size('mb')
        mapper_mask_col = get_mapper_categorical(data, check_cat_length=255)

        #replace categorical before downcasting
        data = data.with_columns(
            [
                pl.col(col).replace(mapping_dict, default=None).cast(pl.UInt64)
                for col, mapping_dict in mapper_mask_col.items()
            ]
        )
        
        mapper_column = get_mapper_numerical(
            data=data, type_mapping_reverse=TYPE_MAPPING_REVERSE,
            dtype_choices=[
                pl.Int16,  pl.Int32, pl.Float32, pl.Int64
            ], categorical_columns=mapper_mask_col.keys()
        )
        #no float64
        mapper_column_cast = {
            col: TYPE_MAPPING[dtype_str]
            for col, dtype_str in mapper_column.items()
        }
        data = data.with_columns(
            [
                pl.col(col).cast(mapper_column_cast[col])
                for col in data.columns
            ]
        )
        size_end = data.estimated_size('mb')
        
        percent = 100 * (size_begin - size_end) / size_begin
        print(
            'Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(
                size_begin, size_end, percent
            )
        )
        mapper_col_all_file[file_name] = mapper_column
        mapper_mask_all_file[file_name] = mapper_mask_col
        mapper_statistic_all_file[file_name] = get_mapper_statistic(data=data)
        
    with open(
        os.path.join(
            config['PATH_MAPPER_DATA'], 'mapper_dtype.json'
        ), 'w'            
    ) as file_dtype:
        
        json.dump(mapper_col_all_file, file_dtype)
        
    with open(
        os.path.join(
            config['PATH_MAPPER_DATA'], 'mapper_mask.json'
        ), 'w'            
    ) as file_mask:
        
        json.dump(mapper_mask_all_file, file_mask)

    with open(
        os.path.join(
            config['PATH_MAPPER_DATA'], 'mapper_statistic.json'
        ), 'w'            
    ) as file_stat:
        
        json.dump(mapper_statistic_all_file, file_stat)