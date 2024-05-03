import os
import gc
import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from functools import partial
from typing import Any
from tqdm import tqdm

from src.base.model.training import ModelTrain
from src.model.xgbm.initialize import XgbInit
from src.model.metric.official_metric import (
    xgb_eval_gini_stability, xgb_slope_part_of_stability
)
 
class XgbTrainer(ModelTrain, XgbInit):
    def _init_train(self) -> None:
        data = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_PARQUET_DATA'],
                'data.parquet'
            )
        )

        excluded_feature = len(self.exclude_feature_list)

        if excluded_feature==0:
            print('Using all feature')
        else:
            print(f'Excluded {excluded_feature} feature')
        
        drop_feature_list: list[str] = (
            self.useless_col_list + 
            [self.fold_name, self.target_col_name] +
            self.exclude_feature_list
        )
        self.feature_list = [
            col for col in data.columns
            if col not in drop_feature_list
        ]
        self.categorical_col_list = [
            col for col in self.categorical_col_list
            if col not in drop_feature_list
        ]
        print(f'Using {len(self.categorical_col_list)} categorical features')

        #save feature list locally for later
        self.save_used_feature()
        self.save_used_categorical_feature()

    def access_fold(self, fold_: int) -> pl.LazyFrame:
        fold_data = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_PARQUET_DATA'],
                'data.parquet'
            )
        )
        fold_data = fold_data.with_columns(
            (
                pl.col('fold_info').str.split(', ')
                .list.get(fold_).alias('current_fold')
            )
        )
        return fold_data

    def train(self) -> None:
        
        self._init_train()
        
        for fold_ in range(self.n_fold):
            print(f'\n\nStarting fold {fold_}\n\n\n')
            
            progress = {}

            print('Collecting dataset')
            fold_data = self.access_fold(fold_=fold_)
            
            train_filtered = fold_data.filter(
                (pl.col('current_fold') == 't')
            )
            test_filtered = fold_data.filter(
                (pl.col('current_fold') == 'v')
            )
            
            assert len(
                set(
                    train_filtered.select('date_decision').unique().collect().to_series().to_list()
                ).intersection(
                    test_filtered.select('date_decision').unique().collect().to_series().to_list()
                )
            ) == 0
            
            train_rows = train_filtered.select(pl.count()).collect().item()
            test_rows = test_filtered.select(pl.count()).collect().item()
            
            print(f'{train_rows} train rows; {test_rows} test rows; {len(self.feature_list)} feature')
            
            assert self.target_col_name not in self.feature_list
            feature_types_list: list[str] = [
                (
                    'c' if col in self.categorical_col_list
                    else 'q'
                )
                for col in self.feature_list
            ]
            train_matrix = xgb.DMatrix(
                train_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float32'),
                train_filtered.select(self.target_col_name).collect().to_pandas().to_numpy('float32').reshape((-1)),
                feature_names=self.feature_list, enable_categorical=True, feature_types=feature_types_list
            )
            
            test_matrix = xgb.DMatrix(
                test_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float32'),
                test_filtered.select(self.target_col_name).collect().to_pandas().to_numpy('float32').reshape((-1)),
                feature_names=self.feature_list, enable_categorical=True, feature_types=feature_types_list
            )

            test_metric_df = test_filtered.select(
                ["WEEK_NUM", "target"]
            ).collect().to_pandas()
                        
            print('Start training')
            model = xgb.train(
                params=self.params_xgb,
                dtrain=train_matrix, 
                num_boost_round=self.params_xgb['n_round'],
                evals=[(test_matrix, 'valid')],
                evals_result=progress, verbose_eval=self.log_evaluation,
                custom_metric=partial(xgb_eval_gini_stability, test_metric_df)
            )

            model.save_model(
                os.path.join(
                    self.experiment_path,
                    f'xgb_{fold_}.json'
                )
            )

            self.model_list.append(model)
            self.progress_list.append(progress)

            del train_matrix, test_matrix
            
            _ = gc.collect()
                
    def save_model(self)->None:
        self.save_pickle_model_list()
        self.save_params()
        self.save_progress_list()
            