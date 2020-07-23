# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 21:04:51 2020

@author: thead
"""
import matplotlib.pyplot as plt
from PIL import Image
import numpy
#I guess numpy and PIL work well together?
image = Image.open('lena.png')
print(image)
data = numpy.array(image)
print(type(data))
print(data.shape)
print(data)
redChan = data[:,:,0]
greenChan = data[:,:, 1]
blueChan = data[:, : , 2]
print(redChan.shape)
print(greenChan.shape)
print(blueChan.shape)

dataConcatenated = numpy.dstack((redChan, greenChan, blueChan))#depth stack!!
dataConcatenated = numpy.float32(dataConcatenated).astype(numpy.uint8)

imageOrig = Image.fromarray(dataConcatenated)
print(type(imageOrig))
plt.imshow(imageOrig)

#imageOrig.save("output.png")