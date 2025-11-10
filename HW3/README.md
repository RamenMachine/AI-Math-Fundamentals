# HW3 - PageRank Algorithm & Graph Analysis

## Overview
Implementation of Google's PageRank algorithm using matrix operations and iterative methods. This project demonstrates the mathematical foundations behind web search ranking systems.

## Files
- `pweek3.py` - Complete PageRank implementation with three core functions

## Key Concepts
- **Matrix Normalization** - Converting adjacency matrices to column-stochastic form
- **Power Iteration** - Iterative method for computing dominant eigenvectors
- **Ranking Systems** - Sorting and organizing results by importance scores
- **Graph Theory** - Understanding web link structures as directed graphs

## Functions Implemented

### 1. Normalize(A)
Normalizes an adjacency matrix so each column sums to 1, creating a transition probability matrix.
- Input: n×n adjacency matrix with 0s and 1s
- Output: n×n normalized matrix (column-stochastic)

### 2. PageRank(L, iter=100)
Computes PageRank vectors through iterative matrix multiplication using the power method.
- Input: Normalized matrix L, number of iterations
- Output: List of rank vectors showing convergence over iterations

### 3. SearchResults(r)
Sorts pages by rank from highest to lowest importance.
- Input: Final rank vector from PageRank
- Output: Sorted list of (index, rank_value) tuples

## Usage
```python
import numpy as np
from pweek3 import Normalize, PageRank, SearchResults

# Create adjacency matrix
A = np.array([[0,1,0,0], [1,0,0,1], [1,0,0,1], [1,1,1,0]])

# Normalize the matrix
L = Normalize(A)

# Run PageRank algorithm
rank_vectors = PageRank(L, iter=100)

# Get sorted results
final_ranks = SearchResults(rank_vectors[-1])
print(final_ranks)
```

## Mathematical Background
PageRank models web surfing as a random walk on a graph. The rank vector represents the probability distribution of being at each page after many random steps. The algorithm converges to the dominant eigenvector of the transition matrix.

**Key Properties:**
- Converges to stable rank distribution
- Starting point doesn't affect final ranking
- Number of iterations controls precision
- Works for any directed graph structure