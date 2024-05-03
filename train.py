if __name__=='__main__':
    import argparse
    import warnings

    from src.utils.import_utils import import_config, import_params
    from src.preprocess.pipeline import PreprocessPipeline

    #filter useless warning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='lgb', type=str)

    args = parser.parse_args()

    config_dict = import_config()
    params_model, experiment_name = import_params(model=args.model)
    
    home_credit_preprocessor = PreprocessPipeline(
        config_dict=config_dict, 
        embarko_skip=4
    )
    home_credit_preprocessor.begin_training()
    
    if args.model == 'lgb':
        from src.model.lgbm.pipeline import LgbmPipeline
        trainer = LgbmPipeline(
            experiment_name=experiment_name + "_lgb",
            params_lgb=params_model,
            config_dict=config_dict, data_columns=home_credit_preprocessor.feature_list,
            metric_eval='gini_stability', log_evaluation=50, 
            evaluate_stability=False, evaluate_shap=False
        )
        trainer.train_explain()
    
    elif args.model == 'ctb':
        from src.model.ctb.pipeline import CTBPipeline

        trainer = CTBPipeline(
            experiment_name=experiment_name + "_ctb",
            params_ctb=params_model,
            config_dict=config_dict, data_columns=home_credit_preprocessor.feature_list,
            metric_eval='CTBGiniStability', log_evaluation=1, 
            evaluate_stability=False, evaluate_shap=False
        )
        trainer.train_explain()

    elif args.model == 'xtb':
        raise NotImplementedError
    
    else:
        raise NotImplementedError