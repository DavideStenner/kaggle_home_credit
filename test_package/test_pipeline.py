import os
import unittest
import warnings
from typing import Dict, Any
from src.utils.import_utils import import_config, import_params
from src.model.lgbm.pipeline import LgbmPipeline
from src.preprocess.pipeline import PreprocessPipeline

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        warnings.filterwarnings("ignore", category=UserWarning)

        self.config_dict: Dict[str, Any] = import_config()
       
        _, experiment_name = import_params(model='lgb')
        self.experiment_name: str = experiment_name
        
        self.pipeline_data: PreprocessPipeline = PreprocessPipeline(
            config_dict=self.config_dict, 
            embarko_skip=6
        )
        
        self.trainer: LgbmPipeline = LgbmPipeline(
            experiment_name=experiment_name + "_lgb",
            params_lgb={},
            config_dict=self.config_dict, 
            data_columns=self.pipeline_data.feature_list,
            metric_eval='gini_stability', log_evaluation=50 
        )

    def test_preprocess_activate_inference(self):
        try:
            self.pipeline_data.begin_inference()
        except Exception as e:
            self.fail(e)
    
    def test_preprocess_activate_inference(self):
        try:
            self.trainer.activate_inference()
        except Exception as e:
            self.fail(e)

    def test_preprocess_import(self):
        try:
            self.pipeline_data.test_all_import(n_rows=1000)
        except Exception as e:
            self.fail(e)

    def test_preprocess_pipeline(self):
        try:
            self.pipeline_data.test_all()
        except Exception as e:
            self.fail(e)
    
    def test_blank_dataset_pipeline(self):
        config_dict = self.config_dict
        config_dict['PATH_ORIGINAL_DATA'] = os.path.join(
            config_dict['PATH_TESTING_DATA'],
            'empty_dataset',
        )

        pipeline_data = PreprocessPipeline(
            config_dict=self.config_dict, 
            embarko_skip=6
        )
        try:
            pipeline_data.test_all()
        except Exception as e:
            self.fail(e)
    
    def test_random_dataset_pipeline(self):
        config_dict = self.config_dict
        config_dict['PATH_ORIGINAL_DATA'] = os.path.join(
            config_dict['PATH_TESTING_DATA'],
            'random_dataset',
        )

        pipeline_data = PreprocessPipeline(
            config_dict=self.config_dict, 
            embarko_skip=6
        )
        try:
            pipeline_data.test_all()
        except Exception as e:
            self.fail(e)
        