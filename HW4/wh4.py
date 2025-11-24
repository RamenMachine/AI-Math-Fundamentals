import numpy as np
import matplotlib.pyplot as plt

# Question 4: bestfit line from week4-1.txt


def question4():
    dataMatrix = np.loadtxt("week4-1.txt", delimiter=",")
    xValues = dataMatrix[:, 0]
    yValues = dataMatrix[:, 1]

    designMatrix = np.column_stack((xValues, np.ones_like(xValues)))

    betaVector, residualsArray, rankValue, singularValues = np.linalg.lstsq(
        designMatrix, yValues, rcond=None)
    slopeM, interceptB = betaVector

    print("Question 4: bestfit line")
    print(f"slope = {slopeM:.6f}")
    print(f"intercept = {interceptB:.6f}")
    if residualsArray.size > 0:
        print(f"residuals = {residualsArray[0]:.6f}")
    print()

    xPlot = np.linspace(xValues.min(), xValues.max(), 400)
    yPlot = slopeM * xPlot + interceptB

    plt.figure()
    plt.scatter(xValues, yValues, label="data")
    plt.plot(xPlot, yPlot, label="bestfit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Q4: bestfit line")
    plt.legend()
    plt.grid(True)
    plt.show()


# Question 5: polynomial y = b1*x + b2*x^2 + b3*x^3
def question5():
    xData = np.array([4, 6, 8, 10, 12, 14, 16, 18], dtype=float)
    yData = np.array([1.58, 2.08, 2.5, 2.8, 3.2, 3.3, 3.9, 4.31], dtype=float)

    designMatrix = np.column_stack((xData, xData**2, xData**3))

    print("Question 5a: design matrix")
    print(designMatrix)
    print()

    betaVector, residualsArray, rankValue, singularValues = np.linalg.lstsq(
        designMatrix, yData, rcond=None)
    coeffB1, coeffB2, coeffB3 = betaVector

    print("Question 5b: coefficients")
    print(f"b1 = {coeffB1:.8f}")
    print(f"b2 = {coeffB2:.8f}")
    print(f"b3 = {coeffB3:.8f}")
    if residualsArray.size > 0:
        print(f"residuals = {residualsArray[0]:.8f}")
    print(
        f"model: y = {coeffB1:.6f}*x + {coeffB2:.6f}*x^2 + {coeffB3:.6f}*x^3")
    print()

    xPlot = np.linspace(xData.min(), xData.max(), 400)
    yPlot = coeffB1 * xPlot + coeffB2 * xPlot**2 + coeffB3 * xPlot**3

    plt.figure()
    plt.scatter(xData, yData, label="data")
    plt.plot(xPlot, yPlot, label="cubic fit")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Q5: cubic polynomial")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    question4()
    question5()
