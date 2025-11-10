#Ameen Rahman

import numpy as np

def Normalize(A):
    # grab the input and convert everything to floats
    inputMatrix = A
    floatMatrix = inputMatrix.astype(float)
    matrixShape = floatMatrix.shape
    numRows = matrixShape[0]
    
    # setup empty matrix for normalized values
    numCols = matrixShape[1]
    normalizedMatrix = np.zeros((numRows, numCols), dtype=float)
    
    for columnIndex in range(numCols):
        # pull out each column and sum it up
        currentColumn = floatMatrix[:, columnIndex]
        columnSum = np.sum(currentColumn)
        
        if columnSum != 0:
            # divide by sum so column adds to 1
            normalizedColumn = currentColumn / columnSum
            for rowIndex in range(numRows):
                normalizedMatrix[rowIndex, columnIndex] = normalizedColumn[rowIndex]
        else:
            # keep zeros as is if column is empty
            for rowIndex in range(numRows):
                normalizedMatrix[rowIndex, columnIndex] = currentColumn[rowIndex]
    
    return normalizedMatrix

def PageRank(L, iter=100):
    # figure out how big the matrix is
    matrixShape = L.shape
    matrixSize = matrixShape[0]
    
    # start everyone with equal rank
    initialValue = 1.0 / matrixSize
    rankVector = np.full((matrixSize, 1), initialValue, dtype=float)
    rankVectorList = []
    
    # keep multiplying to spread the rank around
    iterationCounter = 0
    while iterationCounter < iter:
        nextRankVector = np.dot(L, rankVector)
        rankVectorList.append(nextRankVector)
        
        # update for next round
        rankVector = nextRankVector
        iterationCounter = iterationCounter + 1
    
    return rankVectorList

def SearchResults(r):
    # flatten to 1D and make tuple pairs
    flattenedRankVector = r.flatten()
    numElements = len(flattenedRankVector)
    unsortedTupleList = []
    
    # pair each index with its rank value
    for rowIndex in range(numElements):
        currentValue = flattenedRankVector[rowIndex]
        floatValue = float(currentValue)
        tuplePair = (rowIndex, floatValue)
        unsortedTupleList.append(tuplePair)
    
    # sort indices by rank highest to lowest
    sortedIndices = np.argsort(flattenedRankVector)
    reversedIndices = sortedIndices[::-1]
    sortedTupleList = []
    
    # build final sorted list with proper types
    for idx in reversedIndices:
        integerIndex = int(idx)
        valueAtIndex = flattenedRankVector[idx]
        floatValue = float(valueAtIndex)
        
        # pack it up and ship it
        sortedTuple = (integerIndex, floatValue)
        sortedTupleList.append(sortedTuple)
    
    return sortedTupleList
