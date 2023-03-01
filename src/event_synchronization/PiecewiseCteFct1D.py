from __future__ import annotations
from typing import List, Callable, Tuple, TypeVar
import bisect
from event_synchronization.PiecewiseCteFct import PiecewiseCteFct, ValueType

import numpy as np
import pandas as pd

# ValueType = TypeVar('ValueType')

class Segment:
    min: float
    max: float
    def __init__(self, min, max):
        self.min=min
        self.max=max
    def __str__(self):
        return "[{}, {}]".format(self.min, self.max)
    def __repr__(self):
        return "[{}, {}]".format(self.min, self.max)
        
class TimeVal:
    t: float
    val: ValueType
    def __init__(self, t, val):
        self.t = t
        self.val = val

class PiecewiseCteFct1D(PiecewiseCteFct):
    
    #Always sorted by first float
    values : List[TimeVal]
    
    def __init__(self, val):
        # super().__init__(val)
        self.values=[TimeVal(-np.inf, val), TimeVal(np.inf, val)]
    
    def set_value(self, l: Segment, v: ValueType) -> None :
        raise NotImplementedError
    
    
    def increase_loc(self, l: Segment, v: ValueType) -> None:
        s = self.split(l.min)
        e = self.split(l.max)
        for i in range(s, e):
            self.values[i].val=self.values[i].val+v
    
    @staticmethod
    def sup(f_list: List[PiecewiseCteFct1D]) -> PiecewiseCteFct1D:
        n = len(f_list)
        res = PiecewiseCteFct1D(None)
        indices = [0 for _ in range(len(f_list))]
        
        def find_min_t():
            t = min([f.values[index+1].t for (f, index) in zip(f_list, indices)])
            for i in range(n):
                if f_list[i].values[indices[i]+1].t == t:
                    indices[i] = indices[i]+1 
            return t
        
        def get_max_val():
            return max([f.values[index].val for (f, index) in zip(f_list, indices)])
        
        res.values=[TimeVal(-np.inf, get_max_val())]
        curr = 0
        
        while res.values[curr].t!=np.inf:
            t = find_min_t()
            v = get_max_val()
            curr=curr+1
            res.values.append(TimeVal(t, v))
        
        return res
        
    def map(self, func: Callable[[Segment, ValueType], ValueType]) -> None:
        self.values=[TimeVal(tv.t, func(tv.t, tv.val)) for tv in self.values]
    
    def copy(self) -> PiecewiseCteFct1D:
        copy=PiecewiseCteFct1D(None)
        copy.values=[TimeVal(tv.t, tv.val) for tv in self.values] #v.copy() ? #TODO
        return copy
    
    def optimize(self) -> None:
        i=1
        while i< len(self.values):
            if (self.values[i].val == self.values[i-1].val) and  self.values[i].t!= np.inf:
                if hasattr(self.values[i-1].val, "keep_on_eq"):
                    if not self.values[i-1].val.keep_on_eq(self.values[i]):
                        self.values[i-1].val=self.values[i].val
                del self.values[i]
            else:
                i=i+1
    
    def to_list(self) -> List[Tuple[Segment, ValueType]]:
        return [(Segment(self.values[i].t, self.values[i+1].t), self.values[i].val) for i in range(len(self.values)-1)]
    
    
    
    def split(self, t: float):
        i=bisect.bisect_left(self.values, t, key=lambda x:x.t)
        if self.values[i].t!=t:
            self.values.insert(i, TimeVal(t, self.values[i-1].val))
        return i
                
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([(x.t, x.val) for x in self.values[:-1]], columns=["T", "Val"])
    
    def __str__(self):
        return self.to_dataframe().__str__()
    
    def to_string(self):
        return self.to_dataframe().to_string()
    
    def __repr__(self):
        return self.to_dataframe().__repr__()
    
    def max_value(self):
        return max([tv.val for tv in self.values])

if __name__=="__main__":    
    print("Testing with floats")
    s1 = PiecewiseCteFct1D(0)
    s1.increase_loc(Segment(0, 20), 1)
    s1.increase_loc(Segment(5, 15), 1)
    print(s1)
    s2 = PiecewiseCteFct1D(0)
    s2.increase_loc(Segment(3, 8), 1)
    s2.increase_loc(Segment(7, 13), 1)
    s2.increase_loc(Segment(18, 22), 1)
    print(s2)
    s3 = PiecewiseCteFct1D.sup([s1,s2])
    print(s3)
    s3.optimize()
    print(s3)


if __name__=="__main__": 
    
    print("Testing with TMP")  
    
    class Tmp:
        x:bool
        v:float
        def __lt__(self, o):
            return self.v < o.v
        def __eq__(self, o):
            return self.v == o.v
        def keep_on_eq(self, o):
            return True
            raise NotImplementedError
        def __add__(self,other):
            return Tmp(self.v+other.v)
        def __init__(self, v, x = True):
            self.v = v
            self.x = x
            
        def __str__(self):
            return "{}".format(self.v)
        
        def __repr__(self):
            return "{}".format(self.v)
     
    s1 = PiecewiseCteFct1D(Tmp(0))
    print(s1)
    s1.increase_loc(Segment(0, 20), Tmp(1))
    print(s1)
    s1.increase_loc(Segment(5, 15), Tmp(1))
    print(s1)
    s2 = PiecewiseCteFct1D(Tmp(0))
    s2.increase_loc(Segment(3, 8), Tmp(1))
    s2.increase_loc(Segment(7, 13), Tmp(1))
    s2.increase_loc(Segment(18, 22), Tmp(1))
    print(s2)
    s3 = PiecewiseCteFct1D.sup([s1,s2])
    print(s3)
    s3.optimize()
    print(s3)