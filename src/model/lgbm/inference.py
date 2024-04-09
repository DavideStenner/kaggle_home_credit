import os

import numpy as np
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, log_loss

from src.base.model.inference import ModelPredict
from src.model.lgbm.initialize import LgbmInit
from src.model.metric.official_metric import gini_stability

class LgbmInference(ModelPredict, LgbmInit):     
    def load_feature_data(self, data: pl.DataFrame) -> np.ndarray:
        return data.select(self.feature_list).to_pandas().to_numpy(dtype='float32')
        
    def blend_model_predict(self, test_data: pl.DataFrame) -> np.ndarray:        
        test_data = self.load_feature_data(test_data)
        
        prediction_ = np.zeros((test_data.shape[0]), dtype='float64')
        
        for model in self.model_list:
            prediction_ += model.predict(
                test_data,
                num_iteration = self.best_result['best_epoch']
            )/self.n_fold
            
        return prediction_
    
    def predict(self, test_data: pl.DataFrame) -> np.ndarray:
        assert self.inference
        
        prediction_ = self.blend_model_predict(test_data=test_data)
        return prediction_
    
    def ensemble_model_predict(self, test_data: pl.DataFrame) -> np.ndarray:        
        test_data = self.load_feature_data(test_data)
        
        prediction_ = np.zeros((test_data.shape[0]), dtype='float64')
        
        for model in self.model_ensemble_list:
            prediction_ += model.predict(
                test_data,
                num_iteration = self.best_result['best_epoch']
            )/self.n_fold
            
        return prediction_