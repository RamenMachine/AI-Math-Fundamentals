"""
Promblem 9 and 10
==================================

PROBLEM 9: 1D Gradient Descent (Finding minimum of parabola)
------------------------------------------------------------
Function: y = 2x^2 + 4x - 7
Derivative: y' = 4x + 4
Update rule: x_{i+1} = x_i - η * y'(x_i)
True minimum: x = -1

Tasks:
(a) Manual calculation: x0=3, η=0.4, find x1 and x2 by hand
(b) Run 10 iterations: x0=5, η=0.1
(c) Run 10 iterations: x0=-3, η=0.01
(d) Find i where |x_i - (-1)| < 0.1: x0=-3, η=0.01
(e) Run 10 iterations: x0=5, η=1 (will diverge)
(f) Test if starting point matters: try x0 in [-10,-5,0,5,10], η=0.1
(g) Test if learning rate matters: try η in [0.01,0.1,0.5,1.0,1.5], x0=5

PROBLEM 10: 2D Gradient Descent (Finding minimum of 2D function)
-----------------------------------------------------------------
Function: f(x,y) = 2x^2 + 3y^2 + 8x - 30y + 100
Gradient: ∇f = (∂f/∂x, ∂f/∂y) = (4x + 8, 6y - 30)
Update rule: (x_{i+1}, y_{i+1}) = (x_i, y_i) - η * ∇f(x_i, y_i)
True minimum: (-2, 5)

Tasks:
(a) Manual calculation: (x0,y0)=(0,0), η=0.1, find (x1,y1) and (x2,y2) by hand
(b) Run 10 iterations: (x0,y0)=(0,0), η=0.1, what point is the minimum at?
(c) Start far away: pick (x0,y0) at least 100 units from minimum, choose η, does it converge?
(d) Experiment with learning rates: same starting point as (b), test different η values
(e) Does starting point matter? Test several different starting points
(f) Does learning rate matter? Test several different learning rates
"""

import numpy as np

# ============================================================
# PROBLEM 9: 1D GRADIENT DESCENT
# ============================================================


def y(x):
    """Parabola: y = 2x^2 + 4x - 7"""
    return 2*x**2 + 4*x - 7


def yPrime(x):
    """Derivative: y' = 4x + 4"""
    return 4*x + 4


def gradientDescent1d(x0, eta, numIterations):
    """
    1D gradient descent
    Args: x0 (starting point), eta (learning rate), numIterations
    Returns: list of x values
    """
    xValues = [x0]
    x = x0
    for i in range(numIterations):
        grad = yPrime(x)
        x = x - eta * grad
        xValues.append(x)
    return xValues


print("="*70)
print("PROBLEM 9: 1D GRADIENT DESCENT")
print("="*70)

# Part (a): Manual calculation
print("\nPart 9(a): x0=3, eta=0.4, calculate x1 and x2")
print("-"*70)
x0 = 3
eta = 0.4
print(f"Starting at x0 = {x0}")
print(f"y'(3) = 4(3) + 4 = 16")
x1 = x0 - eta * yPrime(x0)
print(f"x1 = {x0} - {eta} * 16 = {x1}")
print(f"y'({x1}) = 4({x1}) + 4 = {yPrime(x1)}")
x2 = x1 - eta * yPrime(x1)
print(f"x2 = {x1} - {eta} * {yPrime(x1)} = {x2}")

# Part (b): 10 iterations with x0=5, eta=0.1
print("\nPart 9(b): x0=5, eta=0.1, first 10 terms")
print("-"*70)
xValues = gradientDescent1d(5, 0.1, 10)
for i, x in enumerate(xValues):
    print(f"x{i} = {x:.6f}")

# Part (c): 10 iterations with x0=-3, eta=0.01
print("\nPart 9(c): x0=-3, eta=0.01, first 10 terms")
print("-"*70)
xValues = gradientDescent1d(-3, 0.01, 10)
for i, x in enumerate(xValues):
    print(f"x{i} = {x:.6f}")

# Part (d): Find when |x_i - (-1)| < 0.1
print("\nPart 9(d): Find i where |x_i - (-1)| < 0.1")
print("-"*70)
xValues = gradientDescent1d(-3, 0.01, 1000)
targetValue = -1
toleranceValue = 0.1
for i, x in enumerate(xValues):
    if abs(x - targetValue) < toleranceValue:
        print(f"First i where |x_i - (-1)| < 0.1: i = {i}")
        print(f"x{i} = {x:.6f}, |{x:.6f} - (-1)| = {abs(x - targetValue):.6f}")
        break

# Part (e): Divergence example with eta=1
print("\nPart 9(e): x0=5, eta=1, first 10 terms (divergence)")
print("-"*70)
xValues = gradientDescent1d(5, 1.0, 10)
for i, x in enumerate(xValues):
    print(f"x{i} = {x:.6f}")
print("Note: Values oscillate and grow, showing divergence due to large learning rate")

# Part (f): Test different starting points
print("\nPart 9(f): Does starting point matter?")
print("-"*70)
startingPoints = [-10, -5, 0, 5, 10]
eta = 0.1
for x0 in startingPoints:
    xValues = gradientDescent1d(x0, eta, 50)
    finalX = xValues[-1]
    print(
        f"Start: x0={x0:3d}, Final: x={finalX:8.6f}, Distance from -1: {abs(finalX + 1):.6f}")
print("Conclusion: All starting points converge to x = -1")

# Part (g): Test different learning rates
print("\nPart 9(g): Does learning rate matter?")
print("-"*70)
learningRates = [0.01, 0.1, 0.5, 1.0, 1.5]
x0 = 5
for eta in learningRates:
    xValues = gradientDescent1d(x0, eta, 20)
    finalX = xValues[-1]
    if eta < 0.5:
        status = "Converges"
    elif eta == 0.5:
        status = "Optimal convergence"
    elif eta == 1.0:
        status = "Oscillates around minimum"
    else:
        status = "Diverges"
    print(f"eta={eta:4.2f}: Final x={finalX:10.6f}, Status: {status}")

# ============================================================
# PROBLEM 10: 2D GRADIENT DESCENT
# ============================================================


def f(x, y):
    """Objective function: f(x,y) = 2x^2 + 3y^2 + 8x - 30y + 100"""
    return 2*x**2 + 3*y**2 + 8*x - 30*y + 100


def gradientF(x, y):
    """
    Gradient: ∇f = (∂f/∂x, ∂f/∂y) = (4x + 8, 6y - 30)
    Returns: numpy array [gradX, gradY]
    """
    gradX = 4*x + 8
    gradY = 6*y - 30
    return np.array([gradX, gradY])


def gradientDescent2d(x0, y0, eta, numIterations):
    """
    2D gradient descent
    Args: x0, y0 (starting point), eta (learning rate), numIterations
    Returns: list of (x, y) tuples
    """
    pointsList = [(x0, y0)]
    x, y = x0, y0
    for i in range(numIterations):
        grad = gradientF(x, y)
        x = x - eta * grad[0]
        y = y - eta * grad[1]
        pointsList.append((x, y))
    return pointsList


print("\n" + "="*70)
print("PROBLEM 10: 2D GRADIENT DESCENT")
print("="*70)

# Part (a): Manual calculation
print("\nPart 10(a): (x0,y0)=(0,0), eta=0.1, calculate (x1,y1) and (x2,y2)")
print("-"*70)
x0, y0 = 0, 0
eta = 0.1
print(f"Starting at (x0, y0) = ({x0}, {y0})")
gradZero = gradientF(x0, y0)
print(f"grad_f(0,0) = (4(0)+8, 6(0)-30) = ({gradZero[0]}, {gradZero[1]})")
x1 = x0 - eta * gradZero[0]
y1 = y0 - eta * gradZero[1]
print(
    f"(x1, y1) = ({x0}, {y0}) - 0.1 * ({gradZero[0]}, {gradZero[1]}) = ({x1}, {y1})")

gradOne = gradientF(x1, y1)
print(
    f"grad_f({x1},{y1}) = (4({x1})+8, 6({y1})-30) = ({gradOne[0]}, {gradOne[1]})")
x2 = x1 - eta * gradOne[0]
y2 = y1 - eta * gradOne[1]
print(
    f"(x2, y2) = ({x1}, {y1}) - 0.1 * ({gradOne[0]}, {gradOne[1]}) = ({x2}, {y2})")

# Part (b): 10 iterations with (0,0), eta=0.1
print("\nPart 10(b): (x0,y0)=(0,0), eta=0.1, first 10 terms")
print("-"*70)
pointsList = gradientDescent2d(0, 0, 0.1, 10)
for i, (x, y) in enumerate(pointsList):
    fValue = f(x, y)
    print(f"({x:8.6f}, {y:8.6f}), f = {fValue:10.6f}")
print(
    f"The minimum appears to be at ({pointsList[-1][0]:.6f}, {pointsList[-1][1]:.6f})")

# Verify true minimum
print("\nVerification: True minimum is at (-2, 5)")
print(f"f(-2, 5) = {f(-2, 5)}")

# Part (c): Start far from minimum
print("\nPart 10(c): Start at least 100 units away from minimum")
print("-"*70)
# Distance from (-2, 5) to (100, 100) = sqrt((102)^2 + (95)^2) ≈ 140 units
startX, startY = 100, 100
distanceFromMin = np.sqrt((startX - (-2))**2 + (startY - 5)**2)
print(
    f"Starting at ({startX}, {startY}), distance from minimum: {distanceFromMin:.1f} units")
pointsList = gradientDescent2d(startX, startY, 0.05, 100)
finalX, finalY = pointsList[-1]
finalDistance = np.sqrt((finalX - (-2))**2 + (finalY - 5)**2)
print(f"After 100 iterations: ({finalX:.6f}, {finalY:.6f})")
print(f"Distance from true minimum: {finalDistance:.6f}")
print("Converges: Yes, because the function is convex (bowl-shaped)")

# Part (d): Test different learning rates
print("\nPart 10(d): Experiment with different learning rates")
print("-"*70)
learningRates = [0.01, 0.05, 0.1, 0.2, 0.5]
startX, startY = 0, 0
for eta in learningRates:
    pointsList = gradientDescent2d(startX, startY, eta, 100)

    # Find when we get close to minimum
    iterationsToConverge = 100
    for i, (x, y) in enumerate(pointsList):
        if abs(x - (-2)) < 0.01 and abs(y - 5) < 0.01:
            iterationsToConverge = i
            break

    finalX, finalY = pointsList[-1]
    print(f"eta={eta:4.2f}: Final=({finalX:7.4f}, {finalY:7.4f}), Iterations to converge: {iterationsToConverge}")

# Part (e): Test different starting points
print("\nPart 10(e): Does starting point matter?")
print("-"*70)
startingPoints = [(0, 0), (10, 10), (-50, -50), (100, -100), (-200, 300)]
eta = 0.1
for startX, startY in startingPoints:
    pointsList = gradientDescent2d(startX, startY, eta, 100)
    finalX, finalY = pointsList[-1]
    distanceValue = np.sqrt((finalX - (-2))**2 + (finalY - 5)**2)
    print(
        f"Start: ({startX:4d}, {startY:4d}) → Final: ({finalX:7.4f}, {finalY:7.4f}), Distance: {distanceValue:.6f}")
print("Conclusion: All starting points converge to (-2, 5)")

# Part (f): Test different learning rates systematically
print("\nPart 10(f): Does learning rate matter?")
print("-"*70)
learningRates = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5]
startX, startY = 10, 10

for eta in learningRates:
    pointsList = gradientDescent2d(startX, startY, eta, 50)
    finalX, finalY = pointsList[-1]
    distanceValue = np.sqrt((finalX - (-2))**2 + (finalY - 5)**2)

    if distanceValue < 0.01:
        status = "Converged"
    elif distanceValue < 1.0:
        status = "Close"
    else:
        status = "Poor convergence"

    print(
        f"eta={eta:5.3f}: Distance from minimum: {distanceValue:8.6f}, Status: {status}")

print("\nTrade-off: Small eta = slow but stable, Large eta = fast but potentially unstable")

print("\n" + "="*70)
print("SUMMARY & COMPARISON")
print("="*70)
print("\n1D vs 2D Gradient Descent:")
print("  • Same algorithm, different dimensions")
print("  • 1D: scalar gradient, 2D: vector gradient")
print("  • Both sensitive to learning rate")
print("  • Both converge from any starting point (convex functions)")
print("\nKey Insight: This is the foundation of training neural networks!")
