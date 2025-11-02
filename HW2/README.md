# HW2 - Python Week 2 Matrix Operations

## Overview
This assignment implements compact matrix manipulation functions using NumPy for MENG 404 - Math Fundamentals for AI Engineers and Data Scientists.

## Files
- `pweek2.py` - Main implementation file containing all required functions
- `python-week2(1) (1).pdf` - Assignment specifications
- `.venv/` - Python virtual environment with NumPy

## Functions Implemented

### Basic Row Operations
- `swapRows(a, l)` - Swap rows l[0] and l[1] in matrix a
- `mulRow(a, r, c)` - Multiply row r by constant c  
- `addMul(a, l, c)` - Row l[1] = row l[1] + c * row l[0]

### Matrix Operations
- `pivot(a, l)` - Pivot on element a[l[0], l[1]], return 0 if pivot is zero
- `rref(a)` - Return (reduced row echelon form, rank) using pivot operations
- `poolMatrix(a, f, type=0)` - Matrix pooling with f×f filter (type 0=max, type 1=average)

## Usage
```powershell
# Run in Python environment
C:/Users/natsu/OneDrive/Desktop/MENG404/AI-Math-Fundamentals/HW2/.venv/Scripts/python.exe

# Import and use functions
from pweek2 import *
import numpy as np

A = np.array([1,2,3,4], dtype=float).reshape(2,2)
result = swapRows(A, [0,1])  # Swap rows 0 and 1
```

## Code Style
- Extremely compact NumPy code
- camelCase variable and function names  
- Vectorized operations preferred over loops
- Pure functions returning new arrays
- No print statements in functions