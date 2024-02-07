from abc import abstractmethod
from src.base.initialize import BaseInit

class BaseImport(BaseInit):

    @abstractmethod
    def scan_all_dataset(self) -> None:
        pass

    @abstractmethod
    def import_all(self) -> None:
        pass