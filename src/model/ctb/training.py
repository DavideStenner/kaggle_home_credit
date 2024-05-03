import os
import gc
import numpy as np
import pandas as pd
import polars as pl
import catboost as cb

from src.base.model.training import ModelTrain
from src.model.ctb.initialize import CTBInit
from src.model.metric.official_metric import (
    CTBGiniStability, CTBGiniSlope
)
 
class CTBTrainer(ModelTrain, CTBInit):
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
            
            train_feature = (
                train_filtered
                .select(self.feature_list)
                .collect().to_pandas()
            )
            train_feature[self.categorical_col_list] = train_feature[self.categorical_col_list].astype(str)
            
            test_feature = (
                test_filtered
                .select(self.feature_list)
                .collect().to_pandas()
            )
            test_feature[self.categorical_col_list] = test_feature[self.categorical_col_list].astype(str)
            
            train_matrix = cb.Pool(
                data=train_feature,
                label=train_filtered.select(self.target_col_name).collect().to_pandas(),
                feature_names=self.feature_list,
                cat_features=self.categorical_col_list,
            )
            
            test_matrix = cb.Pool(
                data=test_feature,
                label=test_filtered.select(self.target_col_name).collect().to_pandas(),
                feature_names=self.feature_list,
                cat_features=self.categorical_col_list,
            )

            train_metric_df = train_filtered.select(
                ["WEEK_NUM", "target"]
            ).collect().to_pandas()

            test_metric_df = test_filtered.select(
                ["WEEK_NUM", "target"]
            ).collect().to_pandas()
            
            print('Start training')
            model = cb.CatBoostClassifier(
                **self.params_ctb,
                allow_writing_files=False, use_best_model=False,
                eval_metric=CTBGiniStability(test_data=test_metric_df, train_data=train_metric_df),
            )
            model.fit(
                train_matrix, 
                eval_set=test_matrix, 
                metric_period=self.log_evaluation,
            )
            model.save_model(
                os.path.join(
                    self.experiment_path,
                    f'ctb_{fold_}.cbm'
                )
            )

            self.model_list.append(model)
            self.progress_list.append(model.get_evals_result())

            del train_matrix, test_matrix
            
            _ = gc.collect()

    def save_model(self)->None:
        self.save_pickle_model_list()
        self.save_params()
        self.save_progress_list()
            