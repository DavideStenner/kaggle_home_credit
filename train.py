if __name__=='__main__':
    import argparse
    import warnings

    from src.utils.import_utils import import_config, import_params
    from src.model.lgbm.pipeline import LgbmPipeline

    #filter useless warning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    parser = argparse.ArgumentParser()
    parser.add_argument('--xgb', action='store_true')
    args = parser.parse_args()

    config_dict = import_config()
    params_model, experiment_name = import_params(model='lgb')
    
    trainer = LgbmPipeline(
        experiment_name=experiment_name + "_lgb",
        params_lgb=params_model,
        config_dict=config_dict,
        metric_eval='gini_stability', log_evaluation=50, 
        evaluate_stability=True, evaluate_shap=True
    )
    trainer.train_explain()
    
    if args.xgb:
        raise NotImplementedError