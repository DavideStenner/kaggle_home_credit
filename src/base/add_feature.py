from abc import abstractmethod

class BaseFeature():

    @abstractmethod
    def create_feature(self) -> None:
        pass

    @abstractmethod
    def merge_all(self) -> None:
        pass