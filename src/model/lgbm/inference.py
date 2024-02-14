import numpy as np
import polars as pl

from src.base.model.inference import ModelPredict
from src.model.lgbm.initialize import LgbmInit

class LgbmInference(ModelPredict, LgbmInit):     
    def predict(self, test_data: pl.DataFrame) -> np.ndarray:
        raise NotImplementedError