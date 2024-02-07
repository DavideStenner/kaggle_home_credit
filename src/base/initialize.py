import polars as pl

from typing import Union, Any
from abc import ABC, abstractmethod

class BaseInit(ABC):
    
    @abstractmethod
    def _initialiaze_empty_dataset(self) -> None:
        pass
    
    def _collect_item_utils(self, data: Union[pl.DataFrame, pl.LazyFrame]) -> Any:
        return data.item() if self.inference else data.collect().item()