#Ameen Rahman
# PROBLEM 8: NumPy Eigenvalue Calculations

import numpy as np
from numpy import linalg

# given matrix A
A = np.array([[-2, -4, 2],
              [-2, 1, 2],
              [4, 2, 5]])

print("="*70)
print("PROBLEM 8: EIGENVALUE AND EIGENVECTOR ANALYSIS")
print("="*70)

print("\nGiven Matrix A:")
print(A)

# part a: calculate eigenvalues
print("\n" + "-"*70)
print("PART (a): Calculate Eigenvalues using NumPy")
print("-"*70)

eigenvalues, eigenvectors = linalg.eig(A)
print("Eigenvalues:", eigenvalues)
print("\nAnswer: λ₁ = {:.1f}, λ₂ = {:.1f}, λ₃ = {:.1f}".format(
    eigenvalues[0], eigenvalues[1], eigenvalues[2]))

# part b: calculate eigenvectors
print("\n" + "-"*70)
print("PART (b): Calculate Eigenvectors using NumPy")
print("-"*70)

print("Eigenvectors (each column is an eigenvector):")
print(eigenvectors)

print("\nAnswer:")
for i in range(3):
    print(f"For λ = {eigenvalues[i]:.1f}: v ≈ [{eigenvectors[0,i]:.3f}, {eigenvectors[1,i]:.3f}, {eigenvectors[2,i]:.3f}]ᵀ")

# part c: find integer eigenvector for largest eigenvalue
print("\n" + "-"*70)
print("PART (c): Find Integer Eigenvector for Largest Eigenvalue")
print("-"*70)

# find index of largest eigenvalue
largestIndex = np.argmax(eigenvalues)
largestEigenvalue = eigenvalues[largestIndex]
numpyEigenvector = eigenvectors[:, largestIndex]

print(f"Largest eigenvalue: λ = {largestEigenvalue:.1f}")
print(f"NumPy's normalized eigenvector: {numpyEigenvector}")

# manual calculation to find integer eigenvector
# from the work shown we know v = [1, 6, 16]
integerEigenvector = np.array([1, 6, 16])
print(f"\nInteger eigenvector found by hand: v = {integerEigenvector}")

# verify this is correct by checking Av = λv
verificationResult = A @ integerEigenvector
expectedResult = largestEigenvalue * integerEigenvector

print("\nVerification:")
print(f"A × v = {verificationResult}")
print(f"λ × v = {expectedResult}")
print(f"Match: {np.allclose(verificationResult, expectedResult)}")

# part d: show how numpy normalized the eigenvector
print("\n" + "-"*70)
print("PART (d): Show How NumPy Normalized the Eigenvector")
print("-"*70)

print(f"Our integer eigenvector: v = {integerEigenvector}")

# calculate norm
vectorNorm = np.linalg.norm(integerEigenvector)
print(f"\nCalculate norm: ||v|| = √(1² + 6² + 16²) = √{1**2 + 6**2 + 16**2} = {vectorNorm:.3f}")

# normalize
normalizedVector = integerEigenvector / vectorNorm
print(f"\nNormalize: v̂ = v/||v|| = {normalizedVector}")
print(f"NumPy's eigenvector:     {numpyEigenvector}")

# check if they match (accounting for possible sign flip)
matchesExactly = np.allclose(normalizedVector, numpyEigenvector)
matchesFlipped = np.allclose(normalizedVector, -numpyEigenvector)

if matchesExactly:
    print("\n✓ Our normalized vector exactly matches NumPy's output!")
elif matchesFlipped:
    print("\n✓ Our normalized vector matches NumPy's output (with opposite sign)")
    print("  Note: Both directions represent the same eigenspace")
else:
    print("\nDifference:", normalizedVector - numpyEigenvector)

# verify normalized vector has unit length
normalizedNorm = np.linalg.norm(normalizedVector)
print(f"\nVerify unit length: ||v̂|| = {normalizedNorm:.10f}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("NumPy normalizes all eigenvectors to have unit length (norm = 1).")
print("The integer eigenvector [1, 6, 16]ᵀ and NumPy's eigenvector")
print(f"{numpyEigenvector} point in the same direction,")
print("but NumPy's version has been scaled to length 1.")
print("="*70)
