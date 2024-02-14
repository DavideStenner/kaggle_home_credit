from abc import abstractmethod

class BaseCVFold():

    @abstractmethod
    def create_fold(self) -> None:
        pass        
