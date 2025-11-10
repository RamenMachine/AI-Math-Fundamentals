import numpy as np

def Normalize(A):
    adjacencyMatrix = A.astype(float)
    numRows, numCols = adjacencyMatrix.shape
    
    normalizedMatrix = np.zeros((numRows, numCols), dtype=float)
    
    for colIndex in range(numCols):
        columnSum = np.sum(adjacencyMatrix[:, colIndex])
        if columnSum != 0:
            normalizedMatrix[:, colIndex] = adjacencyMatrix[:, colIndex] / columnSum
        else:
            normalizedMatrix[:, colIndex] = adjacencyMatrix[:, colIndex]
    
    return normalizedMatrix

def PageRank(L, iter=100):
    matrixSize = L.shape[0]
    initialValue = 1.0 / matrixSize
    
    rankVector = np.full((matrixSize, 1), initialValue, dtype=float)
    rankVectorList = [rankVector]
    
    for iterationCount in range(iter):
        nextRankVector = np.dot(L, rankVector)
        rankVectorList.append(nextRankVector)
        rankVector = nextRankVector
    
    return rankVectorList

def SearchResults(r):
    flattenedRankVector = r.flatten()
    numElements = len(flattenedRankVector)
    
    tupleList = [(rowIndex, float(flattenedRankVector[rowIndex])) for rowIndex in range(numElements)]
    
    sortedIndices = np.argsort(flattenedRankVector)[::-1]
    sortedTupleList = [(int(idx), float(flattenedRankVector[idx])) for idx in sortedIndices]
    
    return sortedTupleList
