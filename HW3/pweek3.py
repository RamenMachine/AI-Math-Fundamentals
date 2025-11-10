# Python PageRank Implementation for MENG 404 - Week 3
# File: pweek3.py
# CRITICAL: NO print statements in final submission (causes parsing errors)

"""
Implement PageRank algorithm with three functions using NumPy.
ONLY import numpy - no other external modules allowed.

STYLE REQUIREMENTS:
- ALL variables in camelCase (e.g., adjacencyMatrix, columnSum, rankVector)
- Minimal comments only
- Make code more verbose/longer in some sections while keeping full functionality
- Break down operations into multiple steps instead of one-liners
- Use explicit variable assignments for intermediate calculations
"""

import numpy as np

def Normalize(A):
    inputMatrix = A
    floatMatrix = inputMatrix.astype(float)
    matrixShape = floatMatrix.shape
    numRows = matrixShape[0]
    numCols = matrixShape[1]
    normalizedMatrix = np.zeros((numRows, numCols), dtype=float)
    
    for columnIndex in range(numCols):
        currentColumn = floatMatrix[:, columnIndex]
        columnSum = np.sum(currentColumn)
        
        if columnSum != 0:
            normalizedColumn = currentColumn / columnSum
            for rowIndex in range(numRows):
                normalizedMatrix[rowIndex, columnIndex] = normalizedColumn[rowIndex]
        else:
            for rowIndex in range(numRows):
                normalizedMatrix[rowIndex, columnIndex] = currentColumn[rowIndex]
    
    return normalizedMatrix

def PageRank(L, iter=100):
    matrixShape = L.shape
    matrixSize = matrixShape[0]
    initialValue = 1.0 / matrixSize
    rankVector = np.full((matrixSize, 1), initialValue, dtype=float)
    rankVectorList = [rankVector]
    
    iterationCounter = 0
    while iterationCounter < iter:
        nextRankVector = np.dot(L, rankVector)
        rankVectorList.append(nextRankVector)
        rankVector = nextRankVector
        iterationCounter = iterationCounter + 1
    
    return rankVectorList

def SearchResults(r):
    flattenedRankVector = r.flatten()
    numElements = len(flattenedRankVector)
    unsortedTupleList = []
    
    for rowIndex in range(numElements):
        currentValue = flattenedRankVector[rowIndex]
        floatValue = float(currentValue)
        tuplePair = (rowIndex, floatValue)
        unsortedTupleList.append(tuplePair)
    
    sortedIndices = np.argsort(flattenedRankVector)
    reversedIndices = sortedIndices[::-1]
    sortedTupleList = []
    
    for idx in reversedIndices:
        integerIndex = int(idx)
        valueAtIndex = flattenedRankVector[idx]
        floatValue = float(valueAtIndex)
        sortedTuple = (integerIndex, floatValue)
        sortedTupleList.append(sortedTuple)
    
    return sortedTupleList
