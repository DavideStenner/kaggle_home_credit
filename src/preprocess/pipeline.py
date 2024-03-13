import os
import gc

from typing import Any, Tuple

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
        for dataset in self.used_dataset:
            setattr(
                self, 
                dataset,
                getattr(self, dataset).collect()
            )
        
    def collect_all(self) -> None:
        self.collect_feature()
        self.base_data = self.base_data.collect()

    def test_all(self, n_rows: int = 10_000) -> None:
        print(f'Using testing pipeline with head of {n_rows}')
        
        #for debug
        for dataset in self.used_dataset:
            setattr(
                self, 
                dataset,
                getattr(self, dataset).head(n_rows)
            )
        self.base_data = self.base_data.head(n_rows)
        
        self.collect_all()
    
    @property
    def feature_list(self) -> Tuple[str]:
        self.data = None
        self.create_feature()

        self.merge_all()

        self.add_additional_feature()

        data_columns = self.data.columns
        #reset data schema
        self.data = None
        return data_columns
    
    def preprocess_inference(self) -> None:
        print('Creating feature')
        self.create_feature()

        print('Merging All')
        self.merge_all()

        print('Adding additional feature')
        self.add_additional_feature()
        
        print('Collecting test....')
        self.data = self.data.collect()
        _ = gc.collect()

    def preprocess_train(self) -> None:
        print('Creating feature')
        self.create_feature()

        print('Merging All')
        self.merge_all()

        print('Adding additional feature')
        self.add_additional_feature()
        
        print('Collecting....')
        self.data = self.data.collect()
        _ = gc.collect()
        
        print('Creating fold_info column ...')
        self.create_fold()
        self.save_data()
        
    
    def begin_inference(self) -> None:
        print('Beginning inference')
        
        #reset data
        self.data = None
        self.inference: bool = True
        self.path_file_pattern: str = '*/test_{pattern_file}*.parquet'
        
        self.import_all()
        
    def __call__(self) -> None:
        if self.inference:
            self.preprocess_inference()

        else:
            self.preprocess_train()
