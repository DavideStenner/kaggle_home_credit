from abc import abstractmethod
from src.base.initialize import BaseInit

class BaseCVFold(BaseInit):

    @abstractmethod
    def create_fold(self) -> None:
        pass        
