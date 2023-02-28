from __future__ import annotations
from typing import List, Callable, Tuple, TypeVar, Generic
from PiecewiseCteFct import ValueType, PiecewiseCteFct
import logging

logger=logging.getLogger(__name__)

LocationType = TypeVar('LocationType')
    
class Matching:
    l1: List[float]
    l2: List[float]
    m: List[Tuple[int, int]]


class MatchValue(ValueType):
    val: float
    matching: List[Tuple[int, int]]
    is_approx: bool
    
    def __lt__(self, o):
        # logger.info("o is {}".format(o))
        return self.val < o.val
    def __eq__(self, o):
        return self.val == o.val
    def keep_on_eq(self, o):
        return self.is_approx==False
    def __add__(self,other):
        return MatchValue(self.val+other.val, other.matching+self.matching, self.is_approx or other.is_approx)
    def __init__(self, val, matching, is_approx):
        self.val = val
        self.matching = matching
        self.is_approx= is_approx
        
    def __str__(self):
        return "MV({}, {})".format(self.val, self.is_approx)
    
    def __repr__(self):
        return "MV({}, {})".format(self.val, self.is_approx)
    
    def approx(self, val):
        if self.val < val:
            return MatchValue(val, self.matching, True)
        else:
            return MatchValue(self.val, self.matching, self.is_approx)



def get_optimal_matching_per_location(
    l1: List[float], l2: List[float], 
    PiecewiseCteFctType, 
    m: Callable[[float, float], LocationType]
    ) -> List[Tuple[LocationType, Matching, bool]]:
    
    # Initializing dynamic programmming array
    dynamic_prog=[[None for j in range(len(l2)+1)] for i in range(len(l1)+1)]
    
    # Starting with borders (when one of the two lists is empty)
    for i in range(len(l1)+1):
        dynamic_prog[i][len(l2)] = PiecewiseCteFctType(MatchValue(0, [], False))
    for j in range(len(l2)+1):
        dynamic_prog[len(l1)][j] = PiecewiseCteFctType(MatchValue(0, [], False))

    # Handling recursion parameters
    for i in reversed(range(len(l1))):
        for j in reversed(range(len(l2))):
            logger.info("i={}, j={}".format(i, j))
            # Computing the three cases
            rm_l1 = dynamic_prog[i+1][j]
            rm_l2 = dynamic_prog[i][j+1]
            l_match = (dynamic_prog[i+1][j+1])
            # logger.info("rml1={}, rml2={}, match={}".format(rm_l1.values, rm_l2.values, l_match.values))
            l_match=l_match.copy()
            l_match.increase_loc(m(l1[i], l2[j]), MatchValue(1, [(i,j)], False))
            
            # Getting the result
            res=PiecewiseCteFctType.sup([rm_l1, rm_l2, l_match])
            max_matched= res.max_value()
            res.map(lambda l, v : v.approx(max_matched.val/1.5 -10))
            res.optimize()
            dynamic_prog[i][j] = res
    
    result = dynamic_prog[0][0] 
    print(result.to_string())     
    return [(loc, v.matching, v.is_approx) for loc, v in result.to_list()]


      