import os
import json
import pickle

import pandas as pd
import polars as pl
import lightgbm as lgb

from itertools import chain
from typing import Any, Union, Dict, Tuple
from src.base.model.initialize import ModelInit

class LgbmInit(ModelInit):
    def __init__(self, 
            experiment_name:str, 
            params_lgb: dict[str, Any],
            metric_eval: str,
            config_dict: dict[str, Any], data_columns: Tuple[str],
            log_evaluation:int =1, fold_name: str = 'fold_info'
        ):
        
        self.inference: bool = False
        self.config_dict: dict[str, Any] = config_dict
        
        self.experiment_path: str = os.path.join(
            config_dict['PATH_EXPERIMENT'],
            experiment_name
        )
        self.experiment_insight_path: str = os.path.join(
            self.experiment_path, 'insight'
        )
        self.experiment_insight_feat_imp_path: str = os.path.join(
            self.experiment_insight_path, 'feature_importance'
        )

        self.experiment_insight_train_path: str = os.path.join(
            self.experiment_insight_path, 'training'
        )
        self.experiment_shap_path: str = os.path.join(
            self.experiment_path, 'shap'
        )
        self.metric_eval: str = metric_eval
        self.n_fold: int = config_dict['N_FOLD']
        
        self.target_col_name: str = config_dict['TARGET_COL']
        self.fold_name: str = fold_name

        self.special_column_list: list[str] = config_dict['SPECIAL_COLUMNS']

        self.useless_col_list: list[str] = (
            self.special_column_list +[
                'fold_info', 'current_fold', 'date_order_kfold'
            ]
        )
        self.used_dataset: list[str] = (
            config_dict['DEPTH_0'] + config_dict['DEPTH_1'] + config_dict['DEPTH_2']
        )
        self.special_dataset: list[str] = [
            'tax_registry_1'
        ]
        self.log_evaluation: int = log_evaluation

        self.params_lgb: dict[str, Any] = params_lgb
        
        self.model_list: list[lgb.Booster] = []
        self.model_list_stability: list[lgb.Booster] = []
        
        self.progress_list: list = []
        self.best_result: dict[str, Union[int, float]] = None
        
        self.feature_list: list[str] = []
        
        #feature stability
        self.feature_stability_path: str = os.path.join(
            self.experiment_path, 'feature_stability_importances.xlsx'
        )
        self.feature_importance_path: str = os.path.join(
            self.experiment_insight_feat_imp_path, 'feature_importances.xlsx'
        )
        self.exclude_feature_list: list[str] = []

        self.get_categorical_columns(data_columns=data_columns)
    
    def __convert_feature_name_with_dataset(self, mapper_dict: Dict[str, Union[str, dict, float]]):
        return {
            dataset_name: {
                #add dataset name to column as new feature for duplicated column name
                dataset_name + '_' + column: value_
                for column, value_ in mapping_col.items() 
            }
            for dataset_name, mapping_col in
            mapper_dict.items()
        }
    
    def __get_categorical_columns_list(self, mapper_mask: Dict[str, Union[str, dict, float]]) -> list[str]:
        cat_list_col = [
            (
                [
                    dataset_name + '_' + col
                    for col in col_mapper.keys()
                ] +
                [
                    dataset_name + '_' + 'mode_' + col
                    for col in col_mapper.keys()
                ] +
                [
                    dataset_name + '_' + 'not_hashed_missing_mode_' + col
                    for col in col_mapper.keys()
                ]
            )
            for dataset_name, col_mapper in mapper_mask.items()
        ]
        return list(chain(*cat_list_col))
    
    def get_dataset_columns(self) -> None:
        self.load_used_feature()
        
        self.feature_dataset = pd.DataFrame(
            [
                [
                    next((dataset for dataset in ['base'] + self.used_dataset + self.special_dataset if dataset in col)),
                    col
                ]
                for col in self.feature_list
            ],
            columns=['dataset', 'feature']
        )
        
    def get_original_dataset_columns(self) -> None:
        with open(
            os.path.join(
                self.config_dict['PATH_MAPPER_DATA'],
                'mapper_dtype.json'
            ), 'r'
        ) as file:
            mapper_dtype = self.__convert_feature_name_with_dataset(
                json.load(file)
            )
        self.original_feature_dataset = pd.DataFrame(
            list(
                chain(
                    *[
                        [
                            [dataset_name, column]
                            for column in dtype_mapping.keys()
                        ]
                        for dataset_name, dtype_mapping in
                        mapper_dtype.items()
                    ]
                )   
            ),
            columns=[
                'dataset', 'feature'
            ]
        )
        
    def get_categorical_columns(self, data_columns: Tuple[str]) -> None:
        with open(
            os.path.join(
                self.config_dict['PATH_MAPPER_DATA'],
                'mapper_mask.json'
            ), 'r'
        ) as file:
            cat_col_list = self.__get_categorical_columns_list(
                json.load(file)
            )

        self.categorical_col_list: set[str] = (
            set(cat_col_list).intersection(set(data_columns))
        )
        
    def create_experiment_structure(self) -> None:
        for dir_path in [
            self.experiment_path, self.experiment_insight_path, 
            self.experiment_shap_path, self.experiment_insight_train_path,
            self.experiment_insight_feat_imp_path
        ]:
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            
    def load_model(self) -> None: 
        self.load_used_feature()
        self.load_best_result()
        self.load_params()
        self.load_model_list()
        
    def save_progress_list(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'progress_list_lgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(self.progress_list, file)

    def load_progress_list(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'progress_list_lgb.pkl'
            ), 'rb'
        ) as file:
            self.progress_list = pickle.load(file)

    def save_params(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'params_lgb.json'
            ), 'w'
        ) as file:
            json.dump(self.params_lgb, file)
    
    def load_params(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'params_lgb.json'
            ), 'r'
        ) as file:
            self.params_lgb = json.load(file)
    
    def save_best_result(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'best_result_lgb.txt'
            ), 'w'
        ) as file:
            json.dump(self.best_result, file)
        
    def load_best_result(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'best_result_lgb.txt'
            ), 'r'
        ) as file:
            self.best_result = json.load(file)
            
    def save_pickle_model_list(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'model_list_lgb.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(self.model_list, file)
    
    def load_pickle_model_list(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'model_list_lgb.pkl'
            ), 'rb'
        ) as file:
            self.model_list = pickle.load(file)

    def save_pickle_model_stability_list(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'model_list_stability.pkl'
            ), 'wb'
        ) as file:
            pickle.dump(self.model_list_stability, file)
    
    def load_pickle_model_stability_list(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'model_list_stability.pkl'
            ), 'rb'
        ) as file:
            self.model_list_stability = pickle.load(file)

    def load_model_list(self) -> None:
        
        self.model_list = [
            lgb.Booster(
                params=self.params_lgb,
                model_file=os.path.join(
                    self.experiment_path,
                    f'lgb_{fold_}.txt'
                )
            )
            for fold_ in range(self.n_fold)
        ]    
           
    def save_used_feature(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'used_feature.txt'
            ), 'w'
        ) as file:
            json.dump(
                {
                    'feature_model': self.feature_list
                }, 
                file
            )
            
    def load_used_feature(self) -> None:
        with open(
            os.path.join(
                self.experiment_path,
                'used_feature.txt'
            ), 'r'
        ) as file:
            self.feature_list = json.load(file)['feature_model']