import numpy as np
import pandas as pd
import lightgbm as lgb

from typing import Tuple
from sklearn.metrics import roc_auc_score

def gini_stability(
        base: pd.DataFrame,
        w_fallingrate: float=88.0, 
        w_resstd: float=-0.5
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

def custom_eval_gini_stability(
        base: pd.DataFrame, 
        y_pred: np.ndarray, eval_data: lgb.Dataset,
    ) -> Tuple[str, float, bool]:
        base['score'] = y_pred

        eval_result = gini_stability(base=base)

        return 'gini_stability', eval_result, True