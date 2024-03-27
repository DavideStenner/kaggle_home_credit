import os
import sys
sys.path.append(os.getcwd())

if __name__=='__main__':
    import argparse
    import warnings

    import pandas as pd
    
    from src.utils.import_utils import import_config, import_params
    from src.model.lgbm.pipeline import LgbmPipeline
    from src.preprocess.pipeline import PreprocessPipeline

    #filter useless warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--xgb', action='store_true')
    args = parser.parse_args()

    config_dict = import_config()
    params_model, experiment_name = import_params(model='lgb')
    
    class override_fold_eval(PreprocessPipeline):
        def create_fold(self) -> None:
            self.create_time_series_fold()
            
    home_credit_preprocessor = PreprocessPipeline(
        config_dict=config_dict, 
        embarko_skip=6
    )
    home_credit_preprocessor.begin_training()
        
    used_dataset = [
        "credit_bureau_a_1", "static_0", 
        "tax_registry_a_1", "tax_registry_c_1",
        "applprev_1", "static_cb_0", "person_1",
        "tax_registry_b_1", "person_2", "other_1",
        "deposit_1", "debitcard_1", "credit_bureau_b_2",
        "credit_bureau_b_1", "credit_bureau_a_2"
    ]
    result_list = []
    
    for iter_ in range(len(used_dataset)):
        print('Adding ', used_dataset[:iter_])
        
        dataset_list = used_dataset[:iter_]

        home_credit_preprocessor = override_fold_eval(
            config_dict=config_dict, 
            embarko_skip=6
        )
        setattr(
            home_credit_preprocessor, "used_dataset",
            dataset_list
        )

        home_credit_preprocessor._initialize_empty_dataset()
        home_credit_preprocessor._correct_list_date_col()
        home_credit_preprocessor.import_all()
        home_credit_preprocessor.create_feature()
        home_credit_preprocessor.merge_all()
        home_credit_preprocessor.add_additional_feature()
        home_credit_preprocessor.data = home_credit_preprocessor.data.collect()
        home_credit_preprocessor.create_fold()
        home_credit_preprocessor.save_data()

        data_columns = home_credit_preprocessor.data.columns
        
        trainer = LgbmPipeline(
            experiment_name=experiment_name + "_lgb",
            params_lgb=params_model,
            config_dict=config_dict, data_columns=data_columns,
            metric_eval='gini_stability', log_evaluation=50, 
            evaluate_stability=False, evaluate_shap=False
        )
        feature_list = [
            col for col in data_columns
            if col not in (
                trainer.useless_col_list + 
                [trainer.fold_name, trainer.target_col_name] +
                trainer.feature_stability_useless_list
            )
        ]
        print(f'Using {len(feature_list)} feature')

        result = trainer.time_series_fold_train()
        result['dataset'] = used_dataset[iter_]
        result['iteration'] = iter_
        
        result_list.append(result)
    
    result = pd.DataFrame(result_list)
    result.to_excel('time_series_cv.xlsx', index=False)