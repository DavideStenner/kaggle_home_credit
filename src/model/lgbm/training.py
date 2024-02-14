import os
import gc
import polars as pl
import lightgbm as lgb

from functools import partial

from src.base.model.training import ModelTrain
from src.model.lgbm.initialize import LgbmInit
from src.model.metric.official_metric import lgb_eval_gini_stability
 
class LgbmTrainer(ModelTrain, LgbmInit):
    def _init_train(self) -> None:
        data = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_PARQUET_DATA'],
                'data.parquet'
            )
        )

        print('Using all feature')
        self.feature_list = [
            col for col in data.columns
            if col not in self.useless_col_list + [self.fold_name, self.target_col_name]
        ]

        #save feature list locally for later
        self.save_used_feature()
            
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
                (pl.col('current_fold') == 't') &
                (pl.col('target').is_not_null())
            )
            test_filtered = fold_data.filter(
                (pl.col('current_fold') == 'v') &
                (pl.col('target').is_not_null())
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

            metric_lgb = partial(
                lgb_eval_gini_stability, 
                test_filtered.select(
                    ["WEEK_NUM", "target"]
                ).collect().to_pandas()
            )
            
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
                feval=metric_lgb
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

    def all_data_train(self, num_round: int) -> None:
        
        data = pl.scan_parquet(
            os.path.join(
                self.config_dict['PATH_PARQUET_DATA'],
                'data.parquet'
            )
        ).filter(
            (pl.col('target').is_not_null())
        )
        train_matrix = lgb.Dataset(
            data.select(self.feature_list).collect().to_pandas().to_numpy('float32'),
            data.select(self.target_col_name).collect().to_pandas().to_numpy('float64').reshape((-1))
        )
        
        print('Start training model on all data with selected epoch')
        model = lgb.train(
            params=self.params_lgb,
            train_set=train_matrix, 
            feature_name=self.feature_list,
            categorical_feature=self.categorical_col_list,
            num_boost_round=num_round,
        )

        model.save_model(
            os.path.join(
                self.experiment_path,
                f'lgb_all.txt'
            ), importance_type='gain'
        )

    def save_model(self)->None:
        self.save_pickle_model_list()
        self.save_params()
        self.save_progress_list()
            