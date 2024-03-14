# def create_random_value(dataset_: pl.LazyFrame) -> pl.LazyFrame:
#     dtype_mapping = {
#         col: dataset_.dtypes[i]
#         for i, col in enumerate(dataset_.columns)
#     }

#     for col, dtype_ in dtype_mapping.items():
#         if dtype_.is_numeric():
            
        
#         if dataset_.select(col).collect().dty
#         if dtype_mapping[col] != 

def create_testing_dataset():
    
    import os
    import polars as pl
    
    from src.utils.import_utils import import_config
    from src.utils.import_file import read_multiple_parquet
            
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

    if not os.path.isdir(path_blank_test_data):
        os.makedirs(path_blank_test_data)
            
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
        
        empty_special_dataset = (
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

if __name__ == '__main__':
    create_testing_dataset()