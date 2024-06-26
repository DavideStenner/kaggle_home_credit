import os
import gc

from typing import Any, Tuple

from src.base.preprocess.pipeline import BasePipeline
from src.preprocess.import_data import PreprocessImport
from src.preprocess.initialize import PreprocessInit
from src.preprocess.add_feature import PreprocessAddFeature
from src.preprocess.cv_fold import PreprocessFoldCreator
from src.preprocess.filter_feature import PreprocessFilterFeature

class PreprocessPipeline(BasePipeline, PreprocessImport, PreprocessAddFeature, PreprocessFoldCreator, PreprocessFilterFeature):

    def __init__(self, config_dict: dict[str, Any], embarko_skip: int):
                
        PreprocessInit.__init__(
            self, 
            config_dict=config_dict, 
            embarko_skip=embarko_skip
        )

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

    def test_all_import(self, n_rows: int = 10_000) -> None:        
        #for debug
        self.import_all()
        for dataset in self.used_dataset:
            setattr(
                self, 
                dataset,
                getattr(self, dataset).head(n_rows)
            )
        self.base_data = self.base_data.head(n_rows)
        
        self.collect_all()
        
    def test_all(self) -> None:     
        self.import_all()
           
        #for debug
        self.base_data = self.base_data.head(1_000)
        
        self.data = None
        self.create_feature()

        self.merge_all()

        self.add_additional_feature()
        self.data = self.data.collect()
        
    @property
    def feature_list(self) -> Tuple[str]:
        self.data = None
        self.create_feature()

        self.merge_all()

        self.add_additional_feature()

        data_columns = self.data.columns
        #reset data schema
        self.data = None
        
        #reset dataset
        self.import_all()
        
        return data_columns
    
    def preprocess_inference(self, subset_feature: list[str] = None) -> None:
        print('Creating feature')
        self.create_feature()

        print('Merging All')
        self.merge_all()

        print('Adding additional feature')
        self.add_additional_feature()
        
        if subset_feature is not None:
            self.data = self.data.select(subset_feature)
            print(f'Selected {len(subset_feature)} features')
            
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
        
        print(f'Collecting dataset with {len(self.data.columns)} columns')
        self.data = self.data.collect()
        
        #sort by date decision -> no sorting during lgb metric evaluation --> speedup
        self.data = self.data.sort('date_decision', 'WEEK_NUM')
        _ = gc.collect()
        
        print('Creating fold_info column ...')
        self.create_fold()
        self.save_data()
        
        self.filter_feature_by_correlation()
        
    def begin_training(self) -> None:
        self.import_all()
        self.load_excluded_feature()
        
    def begin_inference(self) -> None:
        print('Beginning inference')
        
        #reset data
        self.data = None
        self.inference: bool = True
        self.path_file_pattern: str = '*/test_{pattern_file}*.parquet'
        
        self.import_all()
        
    def __call__(self, subset_feature: list[str] = None) -> None:
        if self.inference:
            self.preprocess_inference(subset_feature=subset_feature)

        else:
            self.import_all()
            self.preprocess_train()
