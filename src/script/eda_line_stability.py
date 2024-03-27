import os
import sys
sys.path.append(os.getcwd())

def get_line_stability() -> None:

    import warnings
    import polars as pl
    
    from src.utils.import_utils import import_config
    from src.preprocess.pipeline import PreprocessPipeline
    from src.eda.stability_feature import save_multiple_line_plot

    warnings.filterwarnings(action='ignore', category=UserWarning)
    
    config_dict = import_config()
    
    if not os.path.isdir('eda/line_stability'):
        os.makedirs('eda/line_stability')

    dataset = pl.read_parquet(
        os.path.join(
            config_dict['PATH_PARQUET_DATA'],
            'data.parquet'
        )
    )
    used_dataset = (
        config_dict['DEPTH_0'] + 
        config_dict['DEPTH_1'] + 
        config_dict['DEPTH_2']
    )
    for dataset_name in used_dataset:
        try:
            print('Running', dataset_name)
            save_multiple_line_plot(dataset=dataset, dataset_name=dataset_name, save_path='eda/line_stability')
            print('Finished', dataset_name)
        except Exception as e:
            print(dataset_name, 'failed...\n\n')
            raise(e)
        
if __name__=='__main__':
    get_line_stability()