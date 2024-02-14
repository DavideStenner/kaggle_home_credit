from abc import abstractmethod

class BasePipeline():    
    @abstractmethod
    def save_data(self) -> None:
        pass
    
    @abstractmethod
    def preprocess_train(self) -> None:
        pass
    
    @abstractmethod
    def collect_feature(self) -> None:
        pass
    
    @abstractmethod
    def collect_all(self) -> None:
        pass