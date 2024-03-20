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
        self.feature_stability_useless_list: list[str] = []
        
        self.get_categorical_columns(data_columns=data_columns)
    
    def convert_feature_name_with_dataset(self, mapper_dict: Dict[str, Union[str, dict, float]]):
        return {
            dataset_name: {
                #add dataset name to column as new feature for duplicated column name
                dataset_name + '_' + column: value_
                for column, value_ in mapping_col.items() 
            }
            for dataset_name, mapping_col in
            mapper_dict.items()
        }
    
    def get_dataset_columns(self) -> None:
        self.load_used_feature()
        
        self.feature_dataset = pd.DataFrame(
            [
                [
                    next((dataset for dataset in self.used_dataset if dataset in col)),
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
            mapper_dtype = self.convert_feature_name_with_dataset(
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
            mapper_mask = self.convert_feature_name_with_dataset(
                json.load(file)
            )

        self.categorical_col_list: set[str] = (
            set(
                list(
                    chain(
                        *[
                            list(type_mapping.keys())
                            for _, type_mapping in mapper_mask.items()
                            if any(type_mapping)
                        ]
                    )
                )
            ).intersection(set(data_columns))
        )
    def create_experiment_structure(self) -> None:
        if not os.path.isdir(self.experiment_path):
            os.makedirs(self.experiment_path)
            
        #plot
        if not os.path.isdir(self.experiment_insight_path):
            os.makedirs(self.experiment_insight_path)
        
        #shap
        if not os.path.isdir(self.experiment_shap_path):
            os.makedirs(self.experiment_shap_path)
            
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