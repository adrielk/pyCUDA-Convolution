# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 20:02:55 2020

@author: thead
"""
import numpy as np
#import pycuda.autoinit
#import pycuda.gpuarray as gpuarray
#import skcuda.fft as cu_fft
from skimage import color
import matplotlib.pyplot as plt
"""
def fft2_gpu(x, fftshift=False):
    
    ''' This function produce an output that is 
    compatible with numpy.fft.fft2
    The input x is a 2D numpy array'''
    # Convert the input array to single precision float
    if x.dtype != 'float32':
        x = x.astype('float32')
    # Get the shape of the initial numpy array
    n1, n2 = x.shape
    
    # From numpy array to GPUarray (I guess what's compatible with pyuda)
    xgpu = gpuarray.to_gpu(x)
    
    # Initialise output GPUarray 
    # For real to complex transformations, the fft function computes 
    # N/2+1 non-redundant coefficients of a length-N input signal.
    y = gpuarray.empty((n1,n2//2 + 1), np.complex64)
    
    # Forward FFT
    # is shape argument meaning it's going to stretch it out to that shape???
    plan_forward = cu_fft.Plan((n1, n2), np.float32, np.complex64)
    cu_fft.fft(xgpu, y, plan_forward)
    #im assuming y is the output array....**
    
    left = y.get()
    # To make the output array compatible with the numpy output
    # we need to stack horizontally the y.get() array and its flipped version
    # We must take care of handling even or odd sized array to get the correct 
    # size of the final array   
    
    if n2//2 == n2/2:
        right = np.roll(np.fliplr(np.flipud(y.get()))[:,1:-1],1,axis=0)
    else:
        right = np.roll(np.fliplr(np.flipud(y.get()))[:,:-1],1,axis=0) 
    
    # Get a numpy array back compatible with np.fft
    if fftshift is False:
        yout = np.hstack((left,right))
    else:
        yout = np.fft.fftshift(np.hstack((left,right)))
    return yout.astype('complex128')
"""
originalImg = plt.imread('rotunda-beginning.jpg')
print(originalImg.dtype)
im = originalImg.astype('uint16')
originalImg = color.rgb2gray(originalImg)
#img = color.r2gray(originalImg)

plt.figure()
plt.imshow(originalImg)

fft1 = np.fft.fftshift(np.fft.fft2(originalImg))
#fft2 = fft2_gpu(img, fftshift = True)
figfft = plt.figure()
plt.imshow(abs(fft1))

imgInverse = np.real(np.fft.ifft2(np.fft.ifftshift(fft1)))
figInverse = plt.figure()
imgInverse= color.gray2rgb(imgInverse)
plt.imshow(imgInverse)

figfft.savefig('fftFig')
figInverse.savefig('figInverse')