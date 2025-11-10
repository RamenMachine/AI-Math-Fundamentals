# pweek1.py - Python Week 1 Assignment

import random
import math
import sumpy
from sympy import *

x = Symbol("x")


def Sigmoid(x):
    return 1 / (1 + math.exp(-x))


def Av(L):
    if len(L) == 0:
        return None

    for element in L:
        if not isinstance(element, (int, float)):
            return None

    return sum(L) / len(L)


def RandAv(n):
    randomList = []
    for i in range(n):
        randomList.append(random.randint(1, n))

    print(randomList)
    return Av(randomList)


def GD(f, x0, n, eta):
    derivative = f.diff(x)
    functionCallable = lambdify(x, f)
    derivativeCallable = lambdify(x, derivative)
    resultList = [x0]

    for i in range(n - 1):
        currentX = resultList[i]
        nextX = currentX - eta * derivativeCallable(currentX)
        resultList.append(nextX)

    return resultList
