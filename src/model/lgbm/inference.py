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

    def time_rescale_prediction(self, data: pl.DataFrame) -> pl.DataFrame:
        data_grouped = (
            data
            .group_by('date_decision')
            .agg(
                pl.col('score').mean()
                .alias('avg_score_by_day')
                .cast(pl.Float32)
            )
            .sort('date_decision')
        )
        
        data_grouped = (
            data_grouped.join(
                (
                    data_grouped
                    .rolling(
                        index_column='date_decision', period="30d", check_sorted=False,
                    ).agg(
                        pl.col('avg_score_by_day').mean().cast(pl.Float32).alias('corrected_score'),
                    )
                ), on='date_decision', how='left'
            )
        ).with_columns(
            (pl.col('corrected_score')/pl.col('avg_score_by_day')).fill_null(1).alias('fix_coef').cast(pl.Float32)
        )
        data = data.join(
            data_grouped.select('date_decision', 'corrected_score'),
            on='date_decision', how='left'
        ).with_columns(
            (pl.col('score')*pl.col('corrected_score')).alias('score').cast(pl.Float32)
        ).drop('corrected_score')
        
        return data

    def evaluate_oof_postprocess_result(self) -> None:
        
        #read data
        oof_prediction = pl.read_parquet(
            os.path.join(self.experiment_path, 'oof_prediction.parquet')
        ).sort('date_decision')
        
        starting_score = gini_stability(
            base=oof_prediction.to_pandas()
        )
        print(f'Initial Stability Gini: {starting_score:.5f}')
        corrected_oof_prediction = self.time_rescale_prediction(data=oof_prediction)
        
        postprocess_score = gini_stability(
            base=corrected_oof_prediction.to_pandas()
        )
        print(f'Final Stability Gini: {postprocess_score:.5f}')

        #score plot
        fig = plt.figure(figsize=(12,8))
        sns.lineplot(
            data=corrected_oof_prediction.filter(
                pl.len().over('date_decision')>30
            ), 
            x="date_decision", y="score", hue='fold'
        )
        plt.title(f"Score prediction over date_decision after post process")
        
        fig.savefig(
            os.path.join(self.experiment_insight_path, 'postprocess_score_over_date.png')
        )
        plt.close(fig)

        #auc over time
        gini_in_time = (
            oof_prediction.to_pandas()
            .sort_values("WEEK_NUM")
            .groupby(["WEEK_NUM", "fold"])[["target", "score"]]
            .apply(
                lambda x: 2*roc_auc_score(x["target"], x["score"])-1
            )
        ).reset_index().rename(columns={0: 'auc'})

        fig = plt.figure(figsize=(12,8))
        sns.lineplot(
            data=gini_in_time, 
            x="WEEK_NUM", y="auc", hue='fold'
        )
        plt.title(f"AUC over WEEK_NUM after post process")
        
        fig.savefig(
            os.path.join(self.experiment_insight_path, 'postprocess_auc_over_week.png')
        )
        plt.close(fig)

        #binary cross entropy over week
        #auc over time
        logloss_in_time = (
            oof_prediction.to_pandas()
            .sort_values("WEEK_NUM")
            .groupby(["WEEK_NUM", "fold"])[["target", "score"]]
            .apply(
                lambda x: log_loss(x["target"], x["score"])
            )
        ).reset_index().rename(columns={0: 'log_loss'})

        fig = plt.figure(figsize=(12,8))
        sns.lineplot(
            data=logloss_in_time, 
            x="WEEK_NUM", y="log_loss", hue='fold'
        )
        plt.title(f"Log Loss over WEEK_NUM after post process")
        
        fig.savefig(
            os.path.join(self.experiment_insight_path, 'postprocess_logloss_over_week.png')
        )
        plt.close(fig)