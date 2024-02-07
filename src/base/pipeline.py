from abc import abstractmethod
from src.base.initialize import BaseInit
from src.base.import_data import BaseImport
from src.base.add_feature import BaseFeature
from src.base.cv_fold import BaseCVFold

class BasePipeline(BaseImport, BaseFeature, BaseCVFold):
    def __init__(self, *args, **kwargs):
        
        BaseInit.__init__(self, *args, **kwargs)
        self.import_all()
    
    @abstractmethod
    def save_data(self) -> None:
        pass
    
    @abstractmethod
    def preprocess_train(self) -> None:
        pass