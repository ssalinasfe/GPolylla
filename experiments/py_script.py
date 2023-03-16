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


numVertices = 7000000
tolerance = 1/numVertices

RandomSample = np.random.rand(numVertices - 2,2)
print("step 1 done!")

RandomSample = add_box(RandomSample, tolerance)

print("step 2 done!")

randomDelaunay = Delaunay(RandomSample)

print("step 3 done!")

randomPoints = RandomSample
randomTriangles = [("triangle", randomDelaunay.simplices)]

import meshio
name = str(numVertices)

meshio.write_points_cells(name+"_uniform.off", randomPoints, randomTriangles)

print("step 4 done!")

import os

folder = "../build"

os.system(folder + "/Polylla "+name+"_uniform.off "+name+"_uniform_out")


print("step 5 done!")
