import os
import gc

from typing import Any

from src.base.preprocess.pipeline import BasePipeline
from src.preprocess.import_data import PreprocessImport
from src.preprocess.initialize import PreprocessInit
from src.preprocess.add_feature import PreprocessAddFeature
from src.preprocess.cv_fold import PreprocessFoldCreator

class PreprocessPipeline(BasePipeline, PreprocessImport, PreprocessAddFeature, PreprocessFoldCreator):

    def __init__(self, config_dict: dict[str, Any], embarko_skip: int):
                
        PreprocessInit.__init__(
            self, 
            config_dict=config_dict, 
            embarko_skip=embarko_skip
        )
        self.import_all()

    def save_data(self) -> None:
        print('saving processed dataset')
        self.data.write_parquet(
            os.path.join(
                self.config_dict['PATH_PARQUET_DATA'],
                'data.parquet'
            )
        )
    def collect_feature(self) -> None:
        self.static_0 = self.static_0.collect()
        self.static_cb_0 = self.static_cb_0.collect()

    def collect_all(self) -> None:
        self.collect_feature()
        self.main_data = self.base_data.collect()

    def preprocess_train(self) -> None:
        self.create_feature()
        self.merge_all()
        
        print('Collecting....')
        self.data = self.data.collect()
        _ = gc.collect()
        
        print('Creating fold_info column ...')
        self.create_fold()
        self.save_data()

    def __call__(self) -> None:
        if self.inference:
            raise NotImplementedError
        else:
            self.preprocess_train()
