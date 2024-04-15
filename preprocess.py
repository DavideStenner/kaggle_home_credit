if __name__=='__main__':
    from src.utils.import_utils import import_config
    from src.preprocess.pipeline import PreprocessPipeline

    config_dict = import_config()
    
    home_credit_preprocessor = PreprocessPipeline(
        config_dict=config_dict, 
        embarko_skip=4
    )
    home_credit_preprocessor()