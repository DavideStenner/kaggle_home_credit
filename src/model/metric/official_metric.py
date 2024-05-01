import numpy as np
import pandas as pd
import lightgbm as lgb

from typing import Tuple
from sklearn.metrics import roc_auc_score

W_FALLINGRATE: float = 88.0
W_RESSTD: float = -.5

def gini_stability(
        base: pd.DataFrame,
        w_fallingrate: float=W_FALLINGRATE, 
        w_resstd: float=W_RESSTD
    ):    
    gini_in_time = (
        base.loc[
            :, 
            ["WEEK_NUM", "target", "score"]
        ]
        .sort_values("WEEK_NUM")
        .groupby("WEEK_NUM")[["target", "score"]]
        .apply(
            lambda x: 2*roc_auc_score(x["target"], x["score"])-1
        )
        .tolist()
    )
    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a*x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    avg_gini = np.mean(gini_in_time)
    return avg_gini + w_fallingrate * min(0, a) + w_resstd * res_std

def lgb_eval_gini_stability(
        base: pd.DataFrame, 
        y_pred: np.ndarray, eval_data: lgb.Dataset,
    ) -> Tuple[str, float, bool]:
        base['score'] = y_pred

        eval_result = gini_stability(base=base)

        return 'gini_stability', eval_result, True

#AVG GINI PART
def avg_gini_part_of_stability(
        base: pd.DataFrame,
        w_fallingrate: float=W_FALLINGRATE, 
        w_resstd: float=W_RESSTD
    ):    
    gini_in_time = (
        base.loc[
            :, 
            ["WEEK_NUM", "target", "score"]
        ]
        .sort_values("WEEK_NUM")
        .groupby("WEEK_NUM")[["target", "score"]]
        .apply(
            lambda x: 2*roc_auc_score(x["target"], x["score"])-1
        )
        .tolist()
    )
    avg_gini = np.mean(gini_in_time)
    return avg_gini

def lgb_avg_gini_part_of_stability(
        base: pd.DataFrame, 
        y_pred: np.ndarray, eval_data: lgb.Dataset,
    ) -> Tuple[str, float, bool]:
        base['score'] = y_pred

        eval_result = avg_gini_part_of_stability(base=base)

        return 'avg_gini', eval_result, True

#SLOPE
def slope_part_of_stability(
        base: pd.DataFrame,
        w_fallingrate: float=W_FALLINGRATE, 
        w_resstd: float=W_RESSTD
    ):    
    gini_in_time = (
        base.loc[
            :, 
            ["WEEK_NUM", "target", "score"]
        ]
        .sort_values("WEEK_NUM")
        .groupby("WEEK_NUM")[["target", "score"]]
        .apply(
            lambda x: 2*roc_auc_score(x["target"], x["score"])-1
        )
        .tolist()
    )
    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    return w_fallingrate * min(0, a)

def lgb_slope_part_of_stability(
        base: pd.DataFrame, 
        y_pred: np.ndarray, eval_data: lgb.Dataset,
    ) -> Tuple[str, float, bool]:
        base['score'] = y_pred

        eval_result = slope_part_of_stability(base=base)

        return 'gini_slope', eval_result, False

#RESIDUAL
def residual_part_of_stability(
        base: pd.DataFrame,
        w_fallingrate: float=W_FALLINGRATE, 
        w_resstd: float=W_RESSTD
    ):    
    gini_in_time = (
        base.loc[
            :, 
            ["WEEK_NUM", "target", "score"]
        ]
        .sort_values("WEEK_NUM")
        .groupby("WEEK_NUM")[["target", "score"]]
        .apply(
            lambda x: 2*roc_auc_score(x["target"], x["score"])-1
        )
        .tolist()
    )
    x = np.arange(len(gini_in_time))
    y = gini_in_time
    a, b = np.polyfit(x, y, 1)
    y_hat = a*x + b
    residuals = y - y_hat
    res_std = np.std(residuals)
    return w_resstd * res_std

def lgb_residual_part_of_stability(
        base: pd.DataFrame, 
        y_pred: np.ndarray, eval_data: lgb.Dataset,
    ) -> Tuple[str, float, bool]:
        base['score'] = y_pred

        eval_result = residual_part_of_stability(base=base)

        return 'gini_residual', eval_result, False

#CATBOOST
class CTBGiniStability():
    def __init__(self, test_data: pd.DataFrame, train_data: pd.DataFrame) -> None:
         self.test_data = test_data
         self.train_data = train_data
         
    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        if approxes[0].shape[0] == self.test_data.shape[0]:
            dataset_ = self.test_data
            dataset_['score'] = approxes[0]
        else:
            dataset_ = self.train_data
            dataset_['score'] = approxes[0]
            
        eval_result = gini_stability(base=dataset_)
        
        return eval_result, 1

    def get_final_error(self, error, weight):
        return error

class CTBGiniSlope():
    def __init__(self, test_data: pd.DataFrame, train_data: pd.DataFrame) -> None:
         self.test_data = test_data
         self.train_data = train_data
         
    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        if approxes[0].shape[0] == self.test_data.shape[0]:
            dataset_ = self.test_data
            dataset_['score'] = approxes[0]
        else:
            dataset_ = self.train_data
            dataset_['score'] = approxes[0]
            
        eval_result = slope_part_of_stability(base=dataset_)
        
        return eval_result, 1

    def get_final_error(self, error, weight):
        return error
