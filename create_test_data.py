def create_testing_dataset():
    
    import os
    import json
    import numpy as np
    import polars as pl
    
    from typing import Dict, Any
    from src.utils.import_utils import import_config
    from src.utils.import_file import read_multiple_parquet
    
    def create_duplicate_on_key(
            dataset_: pl.LazyFrame, 
        ) -> pl.DataFrame:
        
        dataset_: pl.DataFrame = dataset_.collect()

        dataset_ = dataset_.with_columns(
            pl.lit(np.random.randint(1, 10, size=dataset_.height)).alias('quantity_duplicates').cast(pl.UInt8)
        ).select(
            pl.exclude('quantity_duplicates').repeat_by('quantity_duplicates').explode()
        )

        return dataset_

    def create_random_value(dataset_: pl.LazyFrame, dataset_name: str, config_dict: Dict[str, Any]) -> pl.LazyFrame:
        from src.utils.dtype import TYPE_MAPPING
        
        with open(
            os.path.join(
                config_dict['PATH_MAPPER_DATA'],
                'mapper_dtype.json'
            ), 'r'
        ) as file:
            mapper_dtype = json.load(file)
        
        with open(
            os.path.join(
                config_dict['PATH_MAPPER_DATA'],
                'mapper_mask.json'
            ), 'r'
        ) as file:
            categorical_columns = json.load(file)[dataset_name].keys()
            
        for col in dataset_.columns:
            if (col in categorical_columns) | (col in config_dict['SPECIAL_COLUMNS']):
                continue
            
            downcasted_type = mapper_dtype[dataset_name][col]
            doncasted_type_pl = TYPE_MAPPING[downcasted_type]
            
            if doncasted_type_pl.is_numeric():
                
                if 'int' in downcasted_type:
                    dataset_ = dataset_.with_columns(
                        pl.lit(np.random.randint(-1000, 1000)).alias(col).cast(doncasted_type_pl)
                    )
                elif 'float' in downcasted_type:
                    dataset_ = dataset_.with_columns(
                        pl.lit(np.random.random()*2000 - 1000).alias(col).cast(doncasted_type_pl)
                    )
                else:
                    raise ValueError
        
        return dataset_

    config_dict = import_config()
    used_dataset: list[str] = (
        ['base'] +
        config_dict['DEPTH_0'] + config_dict['DEPTH_1'] + config_dict['DEPTH_2']
    )
    path_file_pattern: str = '*/train_{pattern_file}*.parquet'
    
    path_blank_test_data = os.path.join(
        config_dict['PATH_TESTING_DATA'],
        'empty_dataset',
        'train',
    )
    path_random_test_data = os.path.join(
        config_dict['PATH_TESTING_DATA'],
        'random_dataset',
        'train',
    )
    path_repeated_data = os.path.join(
        config_dict['PATH_TESTING_DATA'],
        'duplicated_dataset',
        'train'
    )
    if not os.path.isdir(path_blank_test_data):
        os.makedirs(path_blank_test_data)
    
    if not os.path.isdir(path_random_test_data):
        os.makedirs(path_random_test_data)

    if not os.path.isdir(path_repeated_data):
        os.makedirs(path_repeated_data)

    for pattern_ in  used_dataset:
        print('Saving', pattern_)
        dataset_ = read_multiple_parquet(
            path_file_pattern.format(pattern_file=pattern_),
            root_dir=config_dict['PATH_ORIGINAL_DATA'], 
            scan=True
        ).head(1000)
        
        #retain special col
        special_col_dataset = dataset_.select(
            [
                col for col in dataset_.columns
                if col in config_dict['SPECIAL_COLUMNS']
            ]
        )
        
        other_info = dataset_.select(
            [
                col for col in dataset_.columns
                if col not in config_dict['SPECIAL_COLUMNS']
            ]
        )
        #must create an empty dataset with this information
        empty_dataset = other_info.clear(n=1000)
        
        empty_special_dataset: pl.DataFrame = (
            pl.concat(
                [special_col_dataset, empty_dataset],
                how='horizontal'
            )
        ).collect()

        empty_special_dataset.write_parquet(
            os.path.join(
                path_blank_test_data,
                f'train_{pattern_}.parquet'
            )
        )

        if pattern_ == 'base':
            random_dataset = dataset_
            repeated_dataset = dataset_.collect()
            
        else:
            random_dataset = create_random_value(
                dataset_=dataset_, dataset_name=pattern_, config_dict=config_dict
            )
            repeated_dataset = create_duplicate_on_key(dataset_=dataset_)
            
        random_dataset.sink_parquet(
            os.path.join(
                path_random_test_data,
                f'train_{pattern_}.parquet'
            )
        )
        
        repeated_dataset.write_parquet(
            os.path.join(
                path_repeated_data,
                f'train_{pattern_}.parquet'
            )
        )

if __name__ == '__main__':
    create_testing_dataset()