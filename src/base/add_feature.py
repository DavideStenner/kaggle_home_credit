from abc import abstractmethod
from src.base.initialize import BaseInit

class BaseFeature(BaseInit):

    @abstractmethod
    def create_feature(self) -> None:
        pass