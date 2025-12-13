import numpy as np
#import matplotlib.pyplot as plt




def D(P1,P2):
  return (P1[0]-P2[0])**2 + (P1[1]-P2[1])**2


def Centroid(T):
  # calculate mean of x and y coords
  # return as column vector (2,1)
  return np.mean(T, axis=1).reshape(2,1)




def CreateCentroids(L,K):
  XMin = int(np.min(L[0]))
  XMax = int(np.max(L[0]))
  YMin = int(np.min(L[1]))
  YMax = int(np.max(L[1]))
  #the seed is set here so the program can be tested with an autograder
  #feel free to comment this seed line out in testing, but when submitting to Gradescope, make sure the seed value is set
  np.random.seed(30)
  X = np.random.randint(XMin,XMax,size=(K))
  Y = np.random.randint(YMin,YMax,size=(K))
  return np.vstack((X,Y))
  



def CentroidAssignment(L,C):
  # get number of points and centroids
  N = L.shape[1]
  K = C.shape[1]
  # array to store assignments
  A = np.zeros(N, dtype=int)
  # for each point find closest centroid
  for i in range(N):
    point = L[:, i]
    minDist = float('inf')
    closestCentroid = 0
    # check distance to each centroid
    for c in range(K):
      centroid = C[:, c]
      dist = D(point, centroid)
      # assign to earliest centroid if tie
      if dist < minDist:
        minDist = dist
        closestCentroid = c
    A[i] = closestCentroid
  return A




def NewCentroids(L,A,K):
  # create array to hold new centroids with nan
  newC = np.full((2, K), np.nan)
  # calculate centroid for each cluster
  for k in range(K):
    # find all points assigned to cluster k
    indices = np.where(A == k)[0]
    # get points in this cluster
    clusterPoints = L[:, indices]
    # calculate mean of cluster points
    if clusterPoints.shape[1] > 0:
      newC[:, k] = np.mean(clusterPoints, axis=1)
  return newC





