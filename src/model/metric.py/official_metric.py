import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def gini_stability(
        base: pd.DataFrame, 
        w_fallingrate: float=88.0, 
        w_resstd: float=-0.5
    ):
    assert isinstance(base, pd.DataFrame)
    
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
