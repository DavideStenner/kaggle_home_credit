from abc import abstractmethod

class BaseImport():

    @abstractmethod
    def scan_all_dataset(self) -> None:
        pass

    @abstractmethod
    def import_all(self) -> None:
        pass