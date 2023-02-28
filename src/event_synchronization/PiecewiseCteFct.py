from __future__ import annotations
from typing import List, Callable, Tuple, TypeVar, Generic

LocationType = TypeVar('LocationType')

# Interface for value types of PiecewiseCteFct
class ValueType:
    def __lt__(self, o):
        raise NotImplementedError
    def __eq__(self, o):
        raise NotImplementedError
    def keep_on_eq(self, o): #Optional
        raise NotImplementedError
    def __add__(self,other):
        raise NotImplementedError

# Interface for PiecewiseCteFct
class PiecewiseCteFct:
    
    def __init__(self, val):
        pass
    
    def set_value(self, l: LocationType, v: ValueType) -> None :
        raise NotImplementedError
    
    def increase_loc(self, l: LocationType, v: ValueType) -> None:
        raise NotImplementedError
    
    @staticmethod
    def sup(f_list: List[PiecewiseCteFct]) -> PiecewiseCteFct:
        raise NotImplementedError
        
    def map(self, func: Callable[[LocationType, ValueType], ValueType]) -> None:
        raise NotImplementedError
    
    def copy(self) -> PiecewiseCteFct:
        raise NotImplementedError
    
    def optimize(self) -> None:
        raise NotImplementedError
    
    def to_list(self) -> List[Tuple[LocationType, ValueType]]:
        raise NotImplementedError