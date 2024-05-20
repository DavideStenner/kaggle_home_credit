if __name__=='__main__':
    import argparse
    import warnings

    from src.utils.import_utils import import_config, import_params
    from src.preprocess.pipeline import PreprocessPipeline

    #filter useless warning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='lgb', type=str)
    parser.add_argument('--all_model', action='store_true')
    
    args = parser.parse_args()

    config_dict = import_config()    
    
    home_credit_preprocessor = PreprocessPipeline(
        config_dict=config_dict, 
        embarko_skip=4
    )
    home_credit_preprocessor.begin_training()

    if (args.model == 'ctb') | (args.all_model):
        from src.model.ctb.pipeline import CTBPipeline

        params_model, experiment_name = import_params(model='ctb')

        trainer = CTBPipeline(
            experiment_name=experiment_name + "_ctb",
            params_ctb=params_model,
            config_dict=config_dict, data_columns=home_credit_preprocessor.feature_list,
    
            metric_eval='CTBGiniStability', log_evaluation=1, 
            evaluate_stability=False, evaluate_shap=False
        )
        trainer.train_explain()
        
    if (args.model == 'lgb') | (args.all_model):
        from src.model.lgbm.pipeline import LgbmPipeline

        params_model, experiment_name = import_params(model='lgb')

        trainer = LgbmPipeline(
            experiment_name=experiment_name + "_lgb",
            params_lgb=params_model,
            config_dict=config_dict, data_columns=home_credit_preprocessor.feature_list,
            metric_eval='gini_stability', log_evaluation=50, 
            evaluate_stability=False, evaluate_shap=False
        )
        trainer.train_explain()
    
    if (args.model == 'xgb') | (args.all_model):
        from src.model.xgbm.pipeline import XgbPipeline

        params_model, experiment_name = import_params(model='xgb')

        trainer = XgbPipeline(
            experiment_name=experiment_name + "_xgb",
            params_xgb=params_model,
            config_dict=config_dict, data_columns=home_credit_preprocessor.feature_list,
            metric_eval='gini_stability', log_evaluation=50, 
            evaluate_stability=False, evaluate_shap=False
        )
        trainer.train_explain()