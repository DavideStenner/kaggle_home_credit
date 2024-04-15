import os
import gc
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

from functools import partial
from typing import Any
from tqdm import tqdm

from src.base.model.training import ModelTrain
from src.model.lgbm.initialize import LgbmInit
from src.model.metric.official_metric import (
    lgb_eval_gini_stability, lgb_avg_gini_part_of_stability,
    lgb_residual_part_of_stability, lgb_slope_part_of_stability
)
 
class LgbmTrainer(ModelTrain, LgbmInit):
    def select_model_feature(self) -> None:
        #select only feature which has stability over different folder
        feature_importance: pd.DataFrame = pd.read_excel(
            self.feature_importance_path
        )
        self.exclude_feature_list += feature_importance.loc[feature_importance['average']<100, 'feature'].tolist()
        
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

            callbacks_list = [
                lgb.record_evaluation(progress),
                lgb.log_evaluation(
                    period=self.log_evaluation, 
                    show_stdv=False
                )
            ]
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
            
            train_matrix = lgb.Dataset(
                train_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float32'),
                train_filtered.select(self.target_col_name).collect().to_pandas().to_numpy('float32').reshape((-1))
            )
            
            test_matrix = lgb.Dataset(
                test_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float32'),
                test_filtered.select(self.target_col_name).collect().to_pandas().to_numpy('float32').reshape((-1))
            )

            test_metric_df = test_filtered.select(
                ["WEEK_NUM", "target"]
            ).collect().to_pandas()
            
            metric_lgb_list = [
                partial(function_, test_metric_df)
                for function_ in
                [
                    #add other metric for debug purpose
                    lgb_eval_gini_stability,
                    lgb_slope_part_of_stability
                ]
            ]
            
            print('Start training')
            model = lgb.train(
                params=self.params_lgb,
                train_set=train_matrix, 
                feature_name=self.feature_list,
                categorical_feature=self.categorical_col_list,
                num_boost_round=self.params_lgb['n_round'],
                valid_sets=[test_matrix],
                valid_names=['valid'],
                callbacks=callbacks_list,
                feval=metric_lgb_list
            )

            model.save_model(
                os.path.join(
                    self.experiment_path,
                    f'lgb_{fold_}.txt'
                ), importance_type='gain'
            )

            self.model_list.append(model)
            self.progress_list.append(progress)

            del train_matrix, test_matrix
            
            _ = gc.collect()
        
    def single_fold_train(self) -> None:
        self.load_best_result()
        self.load_used_feature()

        print('\n\n\nBeginning stability training on each validation fold')
        
        for fold_ in range(self.n_fold):
            print(f'\n\nStarting fold {fold_}')
            fold_data = self.access_fold(fold_=fold_)
            
            test_filtered = fold_data.filter(
                (pl.col('current_fold') == 'v')
            )
            test_matrix = lgb.Dataset(
                test_filtered.select(self.feature_list).collect().to_pandas().to_numpy('float32'),
                test_filtered.select(self.target_col_name).collect().to_pandas().to_numpy('float32').reshape((-1))
            )
            
            print('Start training')
            model = lgb.train(
                params=self.params_lgb,
                train_set=test_matrix, 
                feature_name=self.feature_list,
                categorical_feature=self.categorical_col_list,
                num_boost_round=self.best_result['best_epoch'],
            )

            self.model_list_stability.append(model)

            del test_matrix
            
            _ = gc.collect()
        self.save_custom_pickle_model_list(model_list=self.model_list_stability, file_name='model_list_stability.pkl')
        
    def save_model(self)->None:
        self.save_pickle_model_list()
        self.save_params()
        self.save_progress_list()
            