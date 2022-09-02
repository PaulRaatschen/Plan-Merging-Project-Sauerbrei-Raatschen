"""Imports"""
from math import factorial
from enum import Enum
from operator import index
from turtle import position
from typing import Union, List

"""
This file implements a couple of helper methods used for backtrakking by the PrioritizedPlanningSolver. 
"""

def permutation_index(perm : List[int]) -> int:
    """
    Computes the index of a permutation 1..n between 0 and n! - 1.

    Args:
        perm : List with a permutation of numbers. Numbers have to be positive, unique and include all numbers 1..n .
        for the index to be correkt.

    Returns:
        Index of the permutation between 0..n!-1.
    """
    index : int = 0
    size : int = len(perm)
    for i, it in enumerate(perm[:size-1]):
        li : int = 0
        for jt in perm[i+1:]:
            li += it > jt
        index += factorial(size-i-1)*li 
    return index

def index_to_perm(index : int,size : int) -> List[int]:
    """
    Converts an index 1..n!-1 back to the corresponding permutation p(1..n).

    Args:
        index : Index of the permutation.
        size : Length of the permutation.

    Returns:
        List with permutation of numbers 1..n corresponding to index.
    """
    perm : List[int] = [0] * size
    nums : List[int] = list(range(1,size+1))
    for i in range(size):
        size -= 1
        fac : int = factorial(size)
        num : int = index // fac 
        perm[i] = nums[num]
        del nums[num]
        index = index % fac
        
    return perm
    
        
def partial_perm_index(perm : List[int],tsize : int) -> List[int]:
    """
    Computes the range of permutations that begin with a given partial sequence.

    Args:
        perm : Partial permutation with length <= tsize. Numbers have to be positive, unique and inculde a subset of 1..tsize
        for the range to be correct.
        tsize : Length of the complete permutation.

    Returns:
        List where [0] = lower end and [-1] = upper end of the range.
    """
    min_index : int = 0
    delta : int = 0
    psize : int = len(perm)
    
    for i, it in enumerate(perm):
        li : int = 0
        for j in perm[:i]:
            li += it > j
        min_index += (it - 1 - li)*factorial(tsize-i-1)
    
    for i in range(1,tsize-psize):
        delta += i * factorial(i)
    
    return [min_index, min_index + delta] if delta > 0 else [min_index]

class Range(Enum):
    IN = 0
    OVER = 1
    UNDER = 2
    OVERIN = 4
    UNDERIN = 5
    AROUND = 6

def _in_range(r1 : Union[List[int],List[int]], r2 : Union[List[int],List[int]]) -> Range:
    """
    Helper function to check how a range of integers relates to another range.

    Args: 
        r1 : Range of integers or single integer to be compared with r2.
        r2 : Range of integers or single interger with which r1 is compared.

    Returns:
        Range Enum whether r1 is part of r2, overlaps or includes r2.
    """
    r1low : int = r1[0]
    r1up : int = r1[-1]
    r2low : int = r2[0]
    r2up : int = r2[-1]
    
    if r1up < r2low:
        return Range.UNDER
    elif r1low > r2up:
        return Range.OVER
    elif r1up > r2up and r1low < r2low:
        return Range.AROUND
    elif r1up > r2up and r1low >= r2low:
        return Range.OVERIN
    elif r1low < r2low and r1up <= r2up:
        return Range.UNDERIN
    else:
        return Range.IN
    
def update_pos(irange : Union[List[int],List[int]], positions : List[Union[List[int],List[int]]], size : int) -> Union[None, int]:
    """
    Inserts a range of integers or a single integer into a list of integer ranges. If an inserted integer was 
    alerady part of a range, the next closest integer that is not part of a range is returned as well as included in a range.

    Args:
        irange : Range or integer to be inserted.
        positions : List of integers or inter ranges.
        size : Maximum size of an integer in all ranges.

    Side effect:
        If irange was not fully included in a range in positions, positions will be expanded to include irange.
        Else if irange is an integer, the closest integer to irange not included
        will be added to the range that includes irange. 

    Returns:
        None if an integer range was inserted or the inserted integer was not part of a range in positions.
        Else integer that was not part of a range prior to the function call. 
    """
    middle : int = len(positions) // 2
    high : int = len(positions) - 1
    low : int = 0
    index_to_del : List[int] = []
    maximum = factorial(size) - 1
    
    if not positions:
        positions.append(irange)
        return
    
    status : Range = _in_range(irange, positions[middle])
    
    while middle > low and middle < high:
        if status == Range.OVER:
            low = middle
            middle = low + (high - low) // 2 + 1
        elif status == Range.UNDER:
            high = middle
            middle = low + (high-low) // 2
        else:
            break
        status = _in_range(irange, positions[middle])
        
    mrange : List[int] = positions[middle]
        
    if status == Range.UNDER:
        if mrange[0] - irange[-1] == 1:
            if len(mrange) == 2:
                mrange[0] = irange[0]
            else:
                mrange.insert(0,irange[0])
        else:
            positions.insert(middle, irange)
    
    elif status == Range.OVER:
        if irange[0] - mrange[-1] == 1:
            if len(mrange) == 2:
                mrange[-1] = irange[-1]
            else:
                mrange.append(irange[-1])
        else:
            positions.insert(middle+1,irange)
            
    elif status == Range.OVERIN:
        index : int = middle + 1
        while index < len(positions):
            status = _in_range(irange,positions[index])
            if status == Range.UNDER or status == Range.UNDERIN:
                break
            index_to_del.append(index)
            index += 1
        
        if (status == Range.UNDER and positions[index][0] - irange[-1] == 1) or status == Range.UNDERIN:
            index_to_del.append(index)
            if len(mrange) == 2:
                mrange[-1] = positions[index][-1]
            else:
                mrange.append(position[index][-1])
        else :
            if len(mrange) == 2:
                mrange[-1] = irange[-1]
            else:
                mrange.append(irange[-1])
            
        for i in index_to_del:
            del positions[i]
            
    elif status == Range.UNDERIN:
        index : int = middle - 1
        while index >= 0:
            status = _in_range(irange,positions[index])
            if status == Range.OVER or status == Range.OVERIN:
                break
            index_to_del.append(index)
            index -= 1
        
        if (status == Range.OVER and irange[0] - positions[index][-1] == 1) or status == Range.OVERIN:
            index_to_del.append(index)
            if len(mrange) == 2:
                mrange[0] = positions[index][0]
            else:
                mrange.insert(0,positions[index][0])
        else :
            if len(mrange) == 2:
                mrange[0] = irange[0]
            else:
                mrange.insert(0,irange[0])
            
        for i in index_to_del:
            del positions[i]
        
    elif status == Range.AROUND:
        del positions[middle]
        return update_pos(irange,positions,size)
    
    else:
        
        dist : int = (mrange[-1] - irange[-1]) - (irange[0] - mrange[0]) 
        result : int = -1
        
        if dist >= 0 and mrange[0] > 0:
            result : int = mrange[0] - 1
            if middle > 0 and result - positions[middle-1][-1] == 1:
                mrange[0] = positions[middle-1][0]
                del positions[middle-1]       
            else:
                mrange[0] -= 1
                    
        elif mrange[-1] < maximum:
            result : int = mrange[-1] + 1
            if middle < len(positions)-1 and positions[middle+1][0] - result == 1:
                mrange[-1] = positions[middle+1][-1]
                del positions[middle+1]       
            else:
                mrange[-1] += 1
        
        return result