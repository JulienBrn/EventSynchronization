from __future__ import annotations
from typing import List, Callable, Tuple, TypeVar, Generic
from event_synchronization.PiecewiseCteFct import ValueType, PiecewiseCteFct
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
logger=logging.getLogger(__name__)

LocationType = TypeVar('LocationType')
    
class Matching:
    l1: List[float]
    l2: List[float]
    m: List[Tuple[int, int]]
    d: pd.DataFrame()
    def size(self):
        return len(self.m)
    
    @property
    def nb_matched(self):
        return len(self.m)
    
    def __init__(self, l1, l2, m):
        self.l1=l1
        self.l2=l2
        self.m =m
        self.d =self.__to_dataframe()
        
    def to_boolean_lists(self):
        boolL1 = [True if i in [x[0] for x in self.m] else False for i in range(len(self.l1))]
        boolL2 = [True if i in [x[1] for x in self.m] else False for i in range(len(self.l1))]
        return boolL1, boolL2
    
    def to_dataframe(self):
        return self.d.copy()
    def to_matching_dataframe(self):
        return self.d[self.d["l1"].notna() & self.d["l2"].notna()].copy()
    def __to_dataframe(self):
        dm=pd.DataFrame([(i ,j, ind) for ind, (i, j) in enumerate(self.m)], columns=["i", "j", "index"])
        dm["final_index"] = dm["i"] + dm["j"] - dm["index"]
        dl1=pd.DataFrame(enumerate(self.l1), columns=["i", "l1"])
        dl2=pd.DataFrame(enumerate(self.l2), columns=["j", "l2"])
        dl1m = dl1.merge(dm, on="i", how="outer")
        dl1m.sort_values(by=['l1'], inplace=True)
        if pd.isnull(dl1m["final_index"].iat[0]):
            dl1m["final_index"].iat[0]=-1
        dl1m["final_index"].ffill(inplace=True)
        dl1m.loc[dl1m["j"].isna() & dl1m["final_index"].notna(), "final_index"]+=1
    
        res = dl1m.merge(dl2, on="j", how="outer")
        res.sort_values(by=['l2'], inplace=True)
        if pd.isnull(res["final_index"].iat[0]):
            res["final_index"].iat[0]=-1
        res["final_index"].ffill(inplace=True)
        res.loc[res["i"].isna() & res["final_index"].notna(), "final_index"]+=1
        res.sort_values(by=['final_index', "l1"], inplace=True)
        res.reset_index(drop=True, inplace=True)
        res.drop(columns=["i", "j", "final_index", "index"], inplace=True)
        return res
        
    def __str__(self):
        return "{}".format(self.d)

    def __repr__(self):
        return "{}".format(self.d)
    
    def translation_estimate(self) -> float:
        dmatch = self.to_matching_dataframe()
        avg = -(dmatch["l2"] - dmatch["l1"]).mean()
        return avg
    
    def scale_estimate(self) -> float:
        dmatch = self.to_matching_dataframe()
        dmatch["l2_shift"] = dmatch["l2"].shift(1)
        dmatch["l1_shift"] = dmatch["l1"].shift(1)
        dmatch=dmatch.iloc[1:]
        avg_l2 = (dmatch["l2_shift"] - dmatch["l2"]).mean()
        avg_l1 = (dmatch["l1_shift"] - dmatch["l1"]).mean()
        return avg_l1/avg_l2
    
    def lin_transform_estimate(self) -> Tuple[float, float]:
        # shift = self.translation_estimate()
        dmatch = self.to_matching_dataframe()
        scale=self.scale_estimate()
        shift = -(dmatch["l2"]*scale - dmatch["l1"]).mean()
        return scale, shift
    
    def auto_transform(self):
        a,b = self.lin_transform_estimate()
        return Matching(self.l1, [t*a + b for t in self.l2], self.m)
    
    @property
    def nb_unmatched(self):
        return len(self.l1) - self.size(), len(self.l2) - self.size()
    
    def to_string(self):
        return self.d.to_string()
    
    @property
    def total_move(self):
        d = self.to_matching_dataframe()
        return (d["l1"] - d["l2"]).abs().sum()
     
    @property
    def move_per_match(self):
        return self.total_move/float(self.nb_matched)
    
    def plot_matching(self):
        pass
    
    def get_summary(self):
        a,b = self.lin_transform_estimate()
        if abs(a)-1 < 0.0000001 and abs(b) < 0.0000001:
            str = (
                "Summary for matching for l1 of size {} and l2 of size {}\n  "
                    .format(len(self.l1), len(self.l2))
                +"Nb matched: {}, Nb unmatched in l1: {}, Nb unmatched in l2: {}\n  "
                    .format(self.nb_matched, self.nb_unmatched[0], self.nb_unmatched[1])
                + "Matching has already been optimized modulo shifting and scaling\n  "
                +"Total move: {}, Move per matched event: {}\n"
                    .format(self.total_move, self.move_per_match)
            )
        else:
            opt = self.auto_transform()
            str = (
                "Summary for matching for l1 of size {} and l2 of size {}\n  "
                    .format(len(self.l1), len(self.l2))
                +"Nb matched: {}, Nb unmatched in l1: {}, Nb unmatched in l2: {}\n  "
                    .format(self.nb_matched, self.nb_unmatched[0], self.nb_unmatched[1])
                +"Matching has not already been optimized modulo shifting and scaling\n  "
                +"Current total move: {}, current move per matched event: {}\n  "
                    .format(self.total_move, self.move_per_match)
                +"Optimized total move: {}, optimized move per matched event: {}\n  "
                    .format(opt.total_move, opt.move_per_match)
                +"Optimize by calling auto_transform which will shift by {} and scale by {}\n  "
                    .format(self.translation_estimate(), self.scale_estimate())
            )
        return str
    
    def add_matching_eval_plot(self, ax, color="blue"):
        dmatch = self.to_matching_dataframe()
        dmatch["dif"] = dmatch["l1"] - dmatch["l2"]
        ax.ticklabel_format(useOffset=False)
        ax.set_xlabel("i (current match)")
        ax.set_ylabel("l1_matched[i] - l2_matched[i]", color=color)
        # ax.tick_params(axis='y', colors=color)
        return ax.plot([i for i in range(self.nb_matched)], dmatch["dif"].to_list(), color=color)
        
class MatchValue(ValueType):
    val: float
    matching: List[Tuple[int, int]]
    is_approx: bool
    
    def __lt__(self, o):
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
    
    l1 = [float(t) for t in sorted(l1)]
    l2 = [float(t) for t in sorted(l2)]
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
    return [(loc, Matching(l1, l2, v.matching), v.is_approx) for loc, v in result.to_list()]


      