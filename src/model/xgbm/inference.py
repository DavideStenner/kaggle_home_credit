import numpy as np
import polars as pl
import xgboost as xgb

from src.base.model.inference import ModelPredict
from src.model.xgbm.initialize import XgbInit

class XgbInference(ModelPredict, XgbInit):     
    def load_feature_data(self, data: pl.DataFrame) -> xgb.DMatrix:
        feature_data = data.select(self.feature_list).to_pandas().to_numpy(dtype='float32'),
        return xgb.DMatrix(data=feature_data, feature_names=self.feature_list)
        
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