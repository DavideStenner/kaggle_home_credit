import numpy as np
import pandas as pd
import polars as pl

from src.base.model.inference import ModelPredict
from src.model.ctb.initialize import CTBInit

class CTBInference(ModelPredict, CTBInit):     
    def load_feature_data(self, data: pl.DataFrame) -> pd.DataFrame:
        feature_data = data.select(self.feature_list).to_pandas()
        feature_data[self.categorical_col_list] = feature_data[self.categorical_col_list].astype(str)
        return feature_data
        
    def blend_model_predict(self, test_data: pl.DataFrame) -> np.ndarray:        
        test_data = self.load_feature_data(test_data)
        
        prediction_ = np.zeros((test_data.shape[0]), dtype='float64')
        
        for model in self.model_list:
            prediction_ += model.predict(
                test_data, prediction_type='Probability',
                ntree_end = self.best_result['best_epoch']
            )[:, 1]/self.n_fold
            
        return prediction_
    
    def predict(self, test_data: pl.DataFrame) -> np.ndarray:
        assert self.inference
        
        prediction_ = self.blend_model_predict(test_data=test_data)
        return prediction_