import os
import sys
sys.path.append(os.getcwd())

if __name__=='__main__':
    import argparse
    import warnings

    from src.utils.import_utils import import_config, import_params
    from src.model.lgbm.pipeline import LgbmPipeline
    from src.preprocess.pipeline import PreprocessPipeline

    #filter useless warning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('--xgb', action='store_true')
    args = parser.parse_args()

    config_dict = import_config()
    params_model, _ = import_params(model='lgb')
    dataset_list = (
        config_dict['DEPTH_0'] + 
        config_dict['DEPTH_1'] +
        config_dict['DEPTH_2']
    )
    dataset_list = ['credit_bureau_a_2']
    
    home_credit_preprocessor = PreprocessPipeline(
        config_dict=config_dict, 
        embarko_skip=4
    )
    home_credit_preprocessor.begin_training()
    
    for dataset_name in dataset_list:
        print(f'Starting {dataset_name}\n\n')
        
        trainer = LgbmPipeline(
            experiment_name='all_dataset/' + dataset_name + "_lgb",
            params_lgb=params_model,
            config_dict=config_dict, data_columns=home_credit_preprocessor.feature_list,
            metric_eval='gini_stability', log_evaluation=50, 
            evaluate_stability=False, evaluate_shap=False
        )
        trainer.exclude_feature_list = [
            col for col in home_credit_preprocessor.feature_list
            if dataset_name not in col
        ]

        trainer.train_explain()