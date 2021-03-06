# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 00:21:42 2020

@author: thead
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import math 

def low_rank_approx(m, rank = 1):
  U,E,V = np.linalg.svd(m)
  mn = np.zeros_like(m)
  score = 0.0
  for i in range(rank):
    mn += E[i] * np.outer(U[:,i], V[i,:])
    score += E[i]
  print('Approximation percentage, ', score / np.sum(E))
  return mn

plt.style.use('seaborn-whitegrid')
 
x, y = np.meshgrid(np.linspace(-1, 1, 17), np.linspace(-1, 1, 17))
box = np.array(np.logical_and(np.abs(x) < 0.7, np.abs(y) < 0.7),dtype='float64') 
gauss = np.exp(-5 * (x * x + y * y))
#plt.matshow(np.hstack((gauss, box)), cmap='plasma')
#plt.colorbar()
#print(gauss)

#So it might be separable...with approximiation...
bfKernel = np.float64(np.loadtxt('bfKernel.txt'))

U, E, V = np.linalg.svd(bfKernel)
print(E)
UPart = U[:,0]*-1
VPart = V[0,:]*-1
UPart = np.float64(UPart * np.sqrt(E[0]))
VPart = np.float64(VPart * np.sqrt(E[0]))
#bfResult = np.float64(np.outer(UPart, VPart))
print(len(E))
                
bfResult = low_rank_approx(bfKernel, 9)#9 interations, virtually 100% matching(might just be as slow ;(

mse = ((bfResult - bfKernel)**2).mean(axis = None)
print(mse)
#np.savetxt("rowBfKernel.txt", UPart)
#np.savetxt("colBfKernel.txt", VPart)
U2, E2, V2 = np.linalg.svd(bfResult)

plt.matshow(np.hstack((bfKernel, bfResult)), cmap='plasma')
plt.colorbar()

plt.matshow(gauss, cmap='plasma')
plt.colorbar()

"""
-----------------other non perfectly separable kernels
"""
circle = np.array(x*x+y*y < 0.8, dtype='float64')
exponential = np.exp(-3 * np.sqrt(x*x+y*y))
circleApprox = low_rank_approx(circle, rank = 1)
 
hexagon = np.array(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(np.logical_and(x+y < 0.9*np.sqrt(2), x+y > -0.9*np.sqrt(2)),x-y < 0.9*np.sqrt(2)),-x+y < 0.9*np.sqrt(2)),x<0.9),x>-0.9),y<0.9),y>-0.9), dtype='float64')
 
difference_of_gaussians = np.exp(-5 * (x*x+y*y))-np.exp(-6 * (x*x+y*y))
difference_of_gaussians /= np.max(difference_of_gaussians)
 
plt.matshow(np.vstack((np.hstack((circleApprox, hexagon)), np.hstack((exponential, difference_of_gaussians)))), cmap='plasma')
plt.colorbar()


print("Diff of gauss",difference_of_gaussians.shape)

"""
#print(box)
U, E, V = np.linalg.svd(box)
UPart = U[:,0]*-1
VPart = V[0,:]*-1
UPart = UPart * np.sqrt(E[0])
VPart = VPart * np.sqrt(E[0])

#print(E)
print(UPart)
print(VPart)
res = np.outer(UPart, VPart)
print(res)
res = np.float32(res)
box = np.float32(box)
print((res == box).all())
"""

"""
#GAUSS Filter example
#print("Gauss filter: ", gauss)
#print()
U,E,V = np.linalg.svd(gauss)

#print(E)
UPart = U[:,0]*-1
VPart = V[0,:]*-1
UPart = np.multiply(UPart, np.sqrt(E[0]))
VPart = np.multiply(VPart, np.sqrt(E[0]))
print(UPart)
print(VPart)
gauss2 = np.outer(UPart, VPart)
plt.matshow(np.hstack((gauss, gauss2)), cmap='plasma')
plt.colorbar()
#print(gauss)

gauss2 = np.float32(gauss2)
gauss = np.float32(gauss)
print((gauss2 == gauss).all())
"""
