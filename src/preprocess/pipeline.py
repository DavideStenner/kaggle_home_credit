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
        self.person_1 = self.person_1.collect()
        self.applprev_1 = self.applprev_1.collect()
        self.other_1 = self.other_1.collect()
        self.tax_registry_a_1 = self.tax_registry_a_1.collect()
        self.tax_registry_b_1 = self.tax_registry_b_1.collect()
        self.tax_registry_c_1 = self.tax_registry_c_1.collect()

    def collect_all(self) -> None:
        self.collect_feature()
        self.base_data = self.base_data.collect()

    def preprocess_inference(self) -> None:
        self.create_feature()
        self.merge_all()
        
        print('Collecting test....')
        self.data = self.data.collect()
        _ = gc.collect()

    def preprocess_train(self) -> None:
        self.create_feature()
        self.merge_all()
        self.add_additional_feature()
        
        print('Collecting....')
        self.data = self.data.collect()
        _ = gc.collect()
        
        print('Creating fold_info column ...')
        self.create_fold()
        self.save_data()
        
    
    def begin_inference(self) -> None:
        #if no data.parquet is provided create one with 5 rows it's needed by the model to check column names
        if not os.path.exists(
            os.path.join(
                self.config_dict['PATH_PARQUET_DATA'],
                'data.parquet'
            )
        ):
            self.create_feature()
            self.merge_all()
            self.data = self.data.head(5).collect()

            self.save_data()
        
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
