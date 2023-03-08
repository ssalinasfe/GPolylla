# %% [markdown]
# # Experimentos
# 
# Los experimentos se harán en una cuadrilla $[0,1]^2$, para esto, los puntos se intersan aleatoriamente dentro del dominio. 
# 
# Se define la función **move_point** para mover los puntos muy cercanos al borde por un $\epsilon$, y se insertan en el borde cercano
# 
# Luego se eliminan los puntos repetidos y la wea.
# 
# ![image info](https://cdn.shopify.com/s/files/1/0414/9228/3547/files/quienes01.jpg?v=1614317195)

# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from scipy.stats import qmc
from scipy.spatial import Delaunay
import matplotlib.tri as tri

#Aux functions for generate samples
def move_point(max_number, xPoint , yPoint, tolerance):
    r =  np.random.uniform(0, 1)
    n = max_number
    if r > 0.5:
        if xPoint >= max_number*(1.0-tolerance): 
            xPoint = n
        if yPoint >= max_number*(1.0-tolerance): 
            yPoint = n
        if xPoint <= max_number*tolerance: 
            xPoint = 0
        if yPoint <= max_number*tolerance: 
            yPoint = 0
    else:
        if xPoint <= max_number*tolerance: 
            xPoint = 0            
        if yPoint <= max_number*tolerance: 
            yPoint = 0
        if xPoint >= max_number*(1.0-tolerance): 
            xPoint = n
        if yPoint >= max_number*(1.0-tolerance): 
            yPoint = n
    #print("returning", xPoint, yPoint)
    return (xPoint, yPoint)

def add_box(arr, tolerance):
    box = [[0, 0], [1, 1], [0, 1], [1, 0]]
    arr = np.append(arr, box, axis=0)
    np.unique(arr, axis=0)

    maxNumber = max(max(arr[:,0]), max(arr[:,1]))
    for i in range(0, len(arr)):
        new_p = move_point(1, arr[i,0], arr[i,1], tolerance)
        arr[i,0] = new_p[0]
        arr[i,1] = new_p[1]
    return arr

np.random.seed(545)
rng = 4554

# %% [markdown]
# # Random point generation

# %%
startVertice = 1000
stopVertice = 10000
stepVertices = startVertice
tolerance = 0.001

RandomSample = np.random.rand(stopVertice - 4,2)
RandomSample = add_box(RandomSample, tolerance)

import meshio

for i in range(startVertice, stopVertice, stepVertices):
    RandomSubSample = RandomSample[:i]
    # each subSample generates a new triangulation
    randomDelaunay = Delaunay(RandomSubSample)
    randomTriangles =  [("triangle", randomDelaunay.simplices)]
    meshio.write_points_cells(str(len(RandomSubSample)) + "_random.off", RandomSubSample, randomTriangles)


# %% [markdown]
# # Experiment
# 
# In this section we run the benchmark

# %%
import os

folder = "../build"

#os.system(folder + "/Polylla 100_random.off 100_random.out")

for i in range(startVertice, stopVertice, stepVertices):
    os.system(folder + "/Polylla " + str(i) + "_random.off " + str(i) + "random.out")


