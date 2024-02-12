from abc import ABC, abstractmethod

class BaseInit(ABC):
    
    @abstractmethod
    def _initialiaze_empty_dataset(self) -> None:
        pass