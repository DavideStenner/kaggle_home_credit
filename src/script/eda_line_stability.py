import os
import sys
sys.path.append(os.getcwd())

def get_line_stability() -> None:

    import warnings
    
    from src.utils.import_utils import import_config
    from src.preprocess.pipeline import PreprocessPipeline
    from src.eda.stability_feature import save_multiple_line_plot

    warnings.filterwarnings(action='ignore', category=UserWarning)
    
    config_dict = import_config()
    
    if not os.path.isdir('eda/line_stability'):
        os.makedirs('eda/line_stability')

    home_credit_preprocessor = PreprocessPipeline(
        config_dict=config_dict, 
        embarko_skip=6
    )
    #use everything    
    home_credit_preprocessor.import_all()
    home_credit_preprocessor.create_feature()
    home_credit_preprocessor.merge_all()
    home_credit_preprocessor.add_additional_feature()

    dataset = home_credit_preprocessor.data.collect()

    for dataset_name in home_credit_preprocessor.used_dataset:
        try:
            print('Running', dataset_name)
            save_multiple_line_plot(dataset=dataset, dataset_name=dataset_name, save_path='eda/line_stability')
            print('Finished', dataset_name)
        except Exception as e:
            print(dataset_name, 'failed...\n\n')
            raise(e)
        
if __name__=='__main__':
    get_line_stability()