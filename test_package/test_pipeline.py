import os
import unittest
import warnings
from typing import Dict, Any
from src.utils.import_utils import import_config, import_params
from src.model.lgbm.pipeline import LgbmPipeline
from src.model.ctb.pipeline import CTBPipeline
from src.preprocess.pipeline import PreprocessPipeline

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        warnings.filterwarnings("ignore", category=UserWarning)
       
        _, experiment_name = import_params(model='lgb')
        self.experiment_name: str = experiment_name
    
    def test_preprocess_on_train(self):
        try:
            config = import_config()
            pipeline_data: PreprocessPipeline = PreprocessPipeline(
                config_dict=config, 
                embarko_skip=6
            )
            
            pipeline_data.test_all()
            
        except Exception as e:
            self.fail(e)
            
    def test_preprocess_activate_inference(self):
        try:
            config = import_config()
            pipeline_data: PreprocessPipeline = PreprocessPipeline(
                config_dict=config, 
                embarko_skip=6
            )
            
            pipeline_data.begin_inference()
        except Exception as e:
            self.fail(e)
    
    def test_lgb_preprocess_activate_inference(self):
        try:
            config = import_config()
            pipeline_data: PreprocessPipeline = PreprocessPipeline(
                config_dict=config, 
                embarko_skip=6
            )
            pipeline_data.begin_inference()
            
            trainer: LgbmPipeline = LgbmPipeline(
                experiment_name=self.experiment_name + "_lgb",
                params_lgb={},
                config_dict=config, 
                data_columns=pipeline_data.feature_list,
                metric_eval='gini_stability', log_evaluation=50 
            )


            trainer.activate_inference()
        except Exception as e:
            self.fail(e)

    def test_ctb_preprocess_activate_inference(self):
        try:
            config = import_config()
            pipeline_data: PreprocessPipeline = PreprocessPipeline(
                config_dict=config, 
                embarko_skip=6
            )
            pipeline_data.begin_inference()
            
            trainer: CTBPipeline = CTBPipeline(
                experiment_name=self.experiment_name + "_ctb",
                params_ctb={},
                config_dict=config, 
                data_columns=pipeline_data.feature_list,
                metric_eval='CTBGiniStability', log_evaluation=1 
            )


            trainer.activate_inference()
        except Exception as e:
            self.fail(e)

    def test_preprocess_import(self):
        try:
            
            config = import_config()
            pipeline_data: PreprocessPipeline = PreprocessPipeline(
                config_dict=config, 
                embarko_skip=6
            )
            
            pipeline_data.test_all_import(n_rows=1000)
            
        except Exception as e:
            self.fail(e)

    def test_preprocess_pipeline(self):
        try:
            config = import_config()
            pipeline_data: PreprocessPipeline = PreprocessPipeline(
                config_dict=config, 
                embarko_skip=6
            )
            
            pipeline_data.test_all()
        except Exception as e:
            self.fail(e)
    
    def test_blank_dataset_pipeline(self):
        config_dict = import_config()
        config_dict['PATH_ORIGINAL_DATA'] = os.path.join(
            config_dict['PATH_TESTING_DATA'],
            'empty_dataset',
        )

        pipeline_data = PreprocessPipeline(
            config_dict=config_dict, 
            embarko_skip=6
        )
        try:
            pipeline_data.test_all()
        except Exception as e:
            self.fail(e)
    
    def test_random_dataset_pipeline(self):
        config_dict = import_config()
        config_dict['PATH_ORIGINAL_DATA'] = os.path.join(
            config_dict['PATH_TESTING_DATA'],
            'random_dataset',
        )

        pipeline_data = PreprocessPipeline(
            config_dict=config_dict, 
            embarko_skip=6
        )
        try:
            pipeline_data.test_all()
        except Exception as e:
            self.fail(e)

    def test_duplicated_dataset_pipeline(self):
        config_dict = import_config()
        config_dict['PATH_ORIGINAL_DATA'] = os.path.join(
            config_dict['PATH_TESTING_DATA'],
            'duplicated_dataset',
        )

        pipeline_data = PreprocessPipeline(
            config_dict=config_dict, 
            embarko_skip=6
        )
        try:
            pipeline_data.test_all()
        except Exception as e:
            self.fail(e)
    
    def test_missing_dataset_pipeline(self):
        config_dict = import_config()
        config_dict['PATH_ORIGINAL_DATA'] = os.path.join(
            config_dict['PATH_TESTING_DATA'],
            'missing_dataset',
        )

        pipeline_data = PreprocessPipeline(
            config_dict=config_dict, 
            embarko_skip=6
        )
        try:
            pipeline_data.test_all()
        except Exception as e:
            self.fail(e)

    def test_lgb_pipeline_on_test_data(self):
        try:
            config = import_config()
            _, experiment_name = import_params(model='lgb')
            
            pipeline_data = PreprocessPipeline(
                config_dict=config, 
                embarko_skip=6
            )
            pipeline_data.begin_inference()

            trainer = LgbmPipeline(
                experiment_name=experiment_name + "_lgb",
                params_lgb={},
                config_dict=config,
                metric_eval='gini_stability', log_evaluation=50, 
                data_columns=pipeline_data.feature_list
            )
            trainer.activate_inference()
            pipeline_data.test_all()
            
            _ = trainer.predict(pipeline_data.data)

        except Exception as e:
            self.fail(e)

    def test_ctb_pipeline_on_test_data(self):
        try:
            config = import_config()
            _, experiment_name = import_params(model='ctb')
            
            pipeline_data = PreprocessPipeline(
                config_dict=config, 
                embarko_skip=6
            )
            pipeline_data.begin_inference()

            trainer = CTBPipeline(
                experiment_name=experiment_name + "_ctb",
                params_ctb={},
                config_dict=config,
                metric_eval='CTBGiniStability', log_evaluation=1, 
                data_columns=pipeline_data.feature_list
            )
            trainer.activate_inference()
            pipeline_data.test_all()
            
            _ = trainer.predict(pipeline_data.data)

        except Exception as e:
            self.fail(e)