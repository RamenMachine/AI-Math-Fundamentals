# Projected Gradient Descent Implementation
# Optimization on convex sets: unit ball and square box

from sympy import *
import numpy as np
from numpy import linalg as LA

# Define symbolic variables for gradient computation
x = Symbol('x')
y = Symbol('y')


def Proj(P):
    """
    Project point P onto the unit ball D = {(x,y) : x^2 + y^2 <= 1}

    If P is inside or on the ball, return P unchanged.
    If P is outside, return the point on the boundary closest to P.
    """
    # Calculate the norm (distance from origin)
    norm = LA.norm(P)

    # If point is inside or on the unit ball, return unchanged
    if norm <= 1:
        return P

    # Project onto boundary by normalizing to unit length
    return P / norm


def BoxProj(P):
    """
    Project point P onto the square S = {(x,y) : -1 <= x,y <= 1}

    Clips each coordinate independently to the range [-1, 1].
    """
    # Use numpy clip to constrain each coordinate to [-1, 1]
    return np.clip(P, -1, 1)


def PGD(f, P, n, eta, D=1):
    """
    Projected Gradient Descent algorithm for constrained optimization.

    Parameters:
        f: SymPy symbolic function of x and y
        P: Initial point as numpy array [x0, y0]
        n: Number of iterations
        eta: Learning rate (step size)
        D: Domain selector (1=unit ball, 2=square box)

    Returns:
        Array of shape (n+1, 2) containing trajectory [P0, P1, ..., Pn]
    """
    # Compute symbolic partial derivatives
    dfx = f.diff(x)
    dfx = lambdify([x, y], dfx)
    dfy = f.diff(y)
    dfy = lambdify([x, y], dfy)

    # Initialize trajectory list with starting point
    L = np.array([P])
    Temp = P.copy().astype(float)

    # Select projection function based on domain
    projection = Proj if D == 1 else BoxProj

    # Perform n iterations of projected gradient descent
    for i in range(n):
        # Evaluate gradient at current point
        grad_x = dfx(Temp[0], Temp[1])
        grad_y = dfy(Temp[0], Temp[1])
        gradient = np.array([grad_x, grad_y])

        # Take gradient descent step
        Temp = Temp - eta * gradient

        # Project back onto feasible domain
        Temp = projection(Temp)

        # Append to trajectory
        L = np.vstack([L, Temp])

    return L
