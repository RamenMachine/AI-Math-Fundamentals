import numpy as np
from pweek3 import Normalize, PageRank, SearchResults

A = np.array([
    [0, 1, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 0, 1],
    [1, 1, 1, 0]
])

print("Original Matrix A:")
print(A)
print()

normalizedA = Normalize(A)
print("Normalized Matrix:")
print(normalizedA)
print()

rankVectorsList = PageRank(normalizedA, iter=100)
print(f"Number of rank vectors: {len(rankVectorsList)}")
print()

print("Initial rank vector (r0):")
print(rankVectorsList[0])
print()

print("Final rank vector (r100):")
print(rankVectorsList[-1])
print()

searchResults = SearchResults(rankVectorsList[-1])
print("Search Results (sorted by rank):")
for pageIndex, rankValue in searchResults:
    print(f"  Page {pageIndex}: {rankValue:.10f}")
