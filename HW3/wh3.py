#Ameen Rahman
# PROBLEM 8: NumPy Eigenvalue Calculations

import numpy as np
from numpy import linalg

# given matrix A
A = np.array([[-2, -4, 2],
              [-2, 1, 2],
              [4, 2, 5]])

print("="*70)
print("PROBLEM 8: EIGENVALUE AND EIGENVECTOR STUFF")
print("="*70)

print("\nyo heres the matrix A we working with:")
print(A)

# part a: calculate eigenvalues
print("\n" + "-"*70)
print("PART (a): finding eigenvalues with numpy")
print("-"*70)

eigenvalues, eigenvectors = linalg.eig(A)
print("eigenvalues we got:", eigenvalues)
print("\nso basically: λ₁ = {:.1f}, λ₂ = {:.1f}, λ₃ = {:.1f}".format(
    eigenvalues[0], eigenvalues[1], eigenvalues[2]))

# part b: calculate eigenvectors
print("\n" + "-"*70)
print("PART (b): grabbing them eigenvectors")
print("-"*70)

print("eigenvectors (each column is one of em):")
print(eigenvectors)

print("\naight so:")
for i in range(3):
    print(f"when λ = {eigenvalues[i]:.1f}: v is like [{eigenvectors[0,i]:.3f}, {eigenvectors[1,i]:.3f}, {eigenvectors[2,i]:.3f}]ᵀ")

# part c: find integer eigenvector for largest eigenvalue
print("\n" + "-"*70)
print("PART (c): finding clean integer eigenvector for the biggest eigenvalue")
print("-"*70)

# find index of largest eigenvalue
largestIndex = np.argmax(eigenvalues)
largestEigenvalue = eigenvalues[largestIndex]
numpyEigenvector = eigenvectors[:, largestIndex]

print(f"biggest eigenvalue: λ = {largestEigenvalue:.1f}")
print(f"numpys normalized version: {numpyEigenvector}")

# manual calculation to find integer eigenvector
# from the work shown we know v = [1, 6, 16]
integerEigenvector = np.array([1, 6, 16])
print(f"\ninteger eigenvector we found by hand: v = {integerEigenvector}")

# verify this is correct by checking Av = λv
verificationResult = A @ integerEigenvector
expectedResult = largestEigenvalue * integerEigenvector

print("\nlets check if it actually works:")
print(f"A times v = {verificationResult}")
print(f"λ times v = {expectedResult}")
print(f"they match: {np.allclose(verificationResult, expectedResult)}")

# part d: show how numpy normalized the eigenvector
print("\n" + "-"*70)
print("PART (d): showing how numpy made it all normalized and stuff")
print("-"*70)

print(f"our integer eigenvector: v = {integerEigenvector}")

# calculate norm
vectorNorm = np.linalg.norm(integerEigenvector)
print(f"\nfiguring out the norm: ||v|| = √(1² + 6² + 16²) = √{1**2 + 6**2 + 16**2} = {vectorNorm:.3f}")

# normalize
normalizedVector = integerEigenvector / vectorNorm
print(f"\nnow we normalize it: v̂ = v/||v|| = {normalizedVector}")
print(f"numpys eigenvector was:              {numpyEigenvector}")

# check if they match (accounting for possible sign flip)
matchesExactly = np.allclose(normalizedVector, numpyEigenvector)
matchesFlipped = np.allclose(normalizedVector, -numpyEigenvector)

if matchesExactly:
    print("\n✓ yooo our normalized vector is spot on with numpys!")
elif matchesFlipped:
    print("\n✓ our normalized vector matches numpys (just flipped sign)")
    print("  nbd tho both directions work for the eigenspace")
else:
    print("\nuh oh difference:", normalizedVector - numpyEigenvector)

# verify normalized vector has unit length
normalizedNorm = np.linalg.norm(normalizedVector)
print(f"\nchecking unit length: ||v̂|| = {normalizedNorm:.10f}")

print("\n" + "="*70)
print("TLDR")
print("="*70)
print("numpy makes all eigenvectors have length 1 (unit norm).")
print("our integer eigenvector [1, 6, 16]ᵀ and numpys version")
print(f"{numpyEigenvector} are pointing the same way,")
print("numpys just scaled it down to length 1 is all.")
print("="*70)
