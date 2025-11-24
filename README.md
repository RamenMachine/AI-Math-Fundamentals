# Mathematical Foundations for AI/ML

Python implementations of core mathematical algorithms used in machine learning and optimization.

## What's Here

This repo contains implementations of fundamental mathematical techniques - gradient descent, matrix operations, optimization methods, and more. Each folder has working code with comments explaining the math behind it.

## Projects

### HW1 - Gradient Descent
Basic gradient descent implementation with sigmoid activation function. Includes 1D optimization with configurable learning rate and iteration count.

**Key files:** `pweek1.py`

### HW2 - Matrix Operations
Row operations for Gaussian elimination: swap rows, scale rows, add multiples. Also includes RREF computation and max/average pooling for CNN applications.

**Key files:** `pweek2.py`

### HW3 - PageRank Algorithm
Implementation of Google's PageRank using power iteration on column-stochastic matrices. Includes matrix normalization and result ranking.

**Key files:** `pweek3.py`

### HW4 - Least Squares Fitting
Linear and polynomial regression using NumPy's lstsq. Fits lines and cubic polynomials to data with residual analysis.

**Key files:** `wh4.py`

### HW5 - Projected Gradient Descent
Constrained optimization on convex sets (unit ball and square box). Uses SymPy for symbolic differentiation and projects gradient steps back onto feasible regions.

**Key files:** `pweek5.py`

### HW6 - Probability & Statistics
Statistical analysis including chi-square testing and distribution fitting.

### HW7 - Linear Regression
Multi-variable regression analysis with golf score prediction dataset.

### HW8 - Numerical Methods
Finite differences, interpolation, and numerical stability analysis.

### HW9 - Integration
Numerical integration using Simpson's rule and Gaussian quadrature.

## Setup

```bash
pip install numpy scipy matplotlib sympy
```

## Usage

Each homework folder contains standalone Python files. Import and call the functions directly:

```python
from HW5.pweek5 import PGD, Proj, BoxProj
import numpy as np
from sympy import symbols

x, y = symbols('x y')
result = PGD(x**2 + y**2, np.array([3, 4]), 10, 0.1)
```

## Interactive Portfolio

Open `index.html` in a browser to see an interactive visualization of all the mathematical concepts covered in this project.

## Tech Stack

- Python 3.8+
- NumPy for numerical computing
- SymPy for symbolic math
- Matplotlib for visualization
