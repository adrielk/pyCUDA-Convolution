#!python 
'''
/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

/*
 * This sample implements a separable convolution filter
 * of a 2D signal with a gaussian kernel.
 */

 Ported to pycuda by Andrew Wagner <awagner@illinois.edu>, June 2009.
'''

"""
Edited by Adriel Kim:
    1. Can use images and input
    2. New kernels such as sobel and smoothing
"""

import numpy
import time
from PIL import Image
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import string

#for image comparison

def MSEImages(img1, img2):
    diff = (img1-img2)**2
    sumDiff = numpy.sum(diff)
    mse = sumDiff/diff.size
    return mse
    
    

# Pull out a bunch of stuff that was hard coded as pre-processor directives used
# by both the kernel and calling code.
KERNEL_RADIUS = 8#12#1 for a 3x3 kernel
UNROLL_INNER_LOOP = True
KERNEL_W = 2 * KERNEL_RADIUS + 1
ROW_TILE_W = 128
KERNEL_RADIUS_ALIGNED = 16
COLUMN_TILE_W = 16
COLUMN_TILE_H = 48
template = '''
//24-bit multiplication is faster on G80,
//but we must be sure to multiply integers
//only within [-8M, 8M - 1] range
#define IMUL(a, b) __mul24(a, b)

////////////////////////////////////////////////////////////////////////////////
// Kernel configuration
////////////////////////////////////////////////////////////////////////////////
#define KERNEL_RADIUS $KERNEL_RADIUS
#define KERNEL_W $KERNEL_W
__device__ __constant__ float d_Kernel_rows[KERNEL_W];
__device__ __constant__ float d_Kernel_columns[KERNEL_W];

// Assuming ROW_TILE_W, KERNEL_RADIUS_ALIGNED and dataW
// are multiples of coalescing granularity size,
// all global memory operations are coalesced in convolutionRowGPU()
#define            ROW_TILE_W  $ROW_TILE_W
#define KERNEL_RADIUS_ALIGNED  $KERNEL_RADIUS_ALIGNED

// Assuming COLUMN_TILE_W and dataW are multiples
// of coalescing granularity size, all global memory operations
// are coalesced in convolutionColumnGPU()
#define COLUMN_TILE_W $COLUMN_TILE_W
#define COLUMN_TILE_H $COLUMN_TILE_H

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH
){
    //Data cache
    __shared__ float data[KERNEL_RADIUS + ROW_TILE_W + KERNEL_RADIUS];

    //Current tile and apron limits, relative to row start
    const int         tileStart = IMUL(blockIdx.x, ROW_TILE_W);
    const int           tileEnd = tileStart + ROW_TILE_W - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataW - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataW - 1);

    //Row start index in d_Data[]
    const int          rowStart = IMUL(blockIdx.y, dataW);

    //Aligned apron start. Assuming dataW and ROW_TILE_W are multiples
    //of half-warp size, rowStart + apronStartAligned is also a
    //multiple of half-warp size, thus having proper alignment
    //for coalesced d_Data[] read.
    const int apronStartAligned = tileStart - KERNEL_RADIUS_ALIGNED;

    const int loadPos = apronStartAligned + threadIdx.x;
    //Set the entire data cache contents
    //Load global memory values, if indices are within the image borders,
    //or initialize with zeroes otherwise
    if(loadPos >= apronStart){
        const int smemPos = loadPos - apronStart;

        data[smemPos] =
            ((loadPos >= apronStartClamped) && (loadPos <= apronEndClamped)) ?
            d_Data[rowStart + loadPos] : 0;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();
    const int writePos = tileStart + threadIdx.x;
    //Assuming dataW and ROW_TILE_W are multiples of half-warp size,
    //rowStart + tileStart is also a multiple of half-warp size,
    //thus having proper alignment for coalesced d_Result[] write.
    if(writePos <= tileEndClamped){
        const int smemPos = writePos - apronStart;
        float sum = 0;
'''
originalLoop = '''
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += data[smemPos + k] * d_Kernel_rows[KERNEL_RADIUS - k];
'''
unrolledLoop = ''
for k in range(-KERNEL_RADIUS,  KERNEL_RADIUS+1):
    loopTemplate = string.Template(
    'sum += data[smemPos + $k] * d_Kernel_rows[KERNEL_RADIUS - $k];\n')
    unrolledLoop += loopTemplate.substitute(k=k)

#print unrolledLoop
template += unrolledLoop if UNROLL_INNER_LOOP else originalLoop
template += '''
        d_Result[rowStart + writePos] = sum;
        //d_Result[rowStart + writePos] = 128;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH,
    int smemStride,
    int gmemStride
){
    //Data cache
    __shared__ float data[COLUMN_TILE_W *
    (KERNEL_RADIUS + COLUMN_TILE_H + KERNEL_RADIUS)];

    //Current tile and apron limits, in rows
    const int         tileStart = IMUL(blockIdx.y, COLUMN_TILE_H);
    const int           tileEnd = tileStart + COLUMN_TILE_H - 1;
    const int        apronStart = tileStart - KERNEL_RADIUS;
    const int          apronEnd = tileEnd   + KERNEL_RADIUS;

    //Clamp tile and apron limits by image borders
    const int    tileEndClamped = min(tileEnd, dataH - 1);
    const int apronStartClamped = max(apronStart, 0);
    const int   apronEndClamped = min(apronEnd, dataH - 1);

    //Current column index
    const int       columnStart = IMUL(blockIdx.x, COLUMN_TILE_W) + threadIdx.x;

    //Shared and global memory indices for current column
    int smemPos = IMUL(threadIdx.y, COLUMN_TILE_W) + threadIdx.x;
    int gmemPos = IMUL(apronStart + threadIdx.y, dataW) + columnStart;
    //Cycle through the entire data cache
    //Load global memory values, if indices are within the image borders,
    //or initialize with zero otherwise
    for(int y = apronStart + threadIdx.y; y <= apronEnd; y += blockDim.y){
        data[smemPos] =
        ((y >= apronStartClamped) && (y <= apronEndClamped)) ?
        d_Data[gmemPos] : 0;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }

    //Ensure the completness of the loading stage
    //because results, emitted by each thread depend on the data,
    //loaded by another threads
    __syncthreads();
    //Shared and global memory indices for current column
    smemPos = IMUL(threadIdx.y + KERNEL_RADIUS, COLUMN_TILE_W) + threadIdx.x;
    gmemPos = IMUL(tileStart + threadIdx.y , dataW) + columnStart;
    //Cycle through the tile body, clamped by image borders
    //Calculate and output the results
    for(int y = tileStart + threadIdx.y; y <= tileEndClamped; y += blockDim.y){
        float sum = 0;
'''
originalLoop = '''
        for(int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++)
            sum += data[smemPos + IMUL(k, COLUMN_TILE_W)] *
            d_Kernel_columns[KERNEL_RADIUS - k];
'''
unrolledLoop = ''
for k in range(-KERNEL_RADIUS,  KERNEL_RADIUS+1):
    loopTemplate = string.Template('sum += data[smemPos + IMUL($k, COLUMN_TILE_W)] * d_Kernel_columns[KERNEL_RADIUS - $k];\n')
    unrolledLoop += loopTemplate.substitute(k=k)

#print unrolledLoop
template += unrolledLoop if UNROLL_INNER_LOOP else originalLoop
template += '''
        d_Result[gmemPos] = sum;
        //d_Result[gmemPos] = 128;
        smemPos += smemStride;
        gmemPos += gmemStride;
    }
}
'''
template = string.Template(template)
code = template.substitute(KERNEL_RADIUS = KERNEL_RADIUS,
                           KERNEL_W = KERNEL_W,
                           COLUMN_TILE_H=COLUMN_TILE_H,
                           COLUMN_TILE_W=COLUMN_TILE_W,
                           ROW_TILE_W=ROW_TILE_W,
                           KERNEL_RADIUS_ALIGNED=KERNEL_RADIUS_ALIGNED)

module = SourceModule(code)
convolutionRowGPU = module.get_function('convolutionRowGPU')
convolutionColumnGPU = module.get_function('convolutionColumnGPU')
d_Kernel_rows = module.get_global('d_Kernel_rows')[0]
d_Kernel_columns = module.get_global('d_Kernel_columns')[0]

# Helper functions for computing alignment...
def iDivUp(a, b):
    # Round a / b to nearest higher integer value
    a = numpy.int32(a)
    b = numpy.int32(b)
    return (a / b + 1) if (a % b != 0) else (a / b)

def iDivDown(a, b):
    # Round a / b to nearest lower integer value
    a = numpy.int32(a)
    b = numpy.int32(b)
    return a / b;

def iAlignUp(a, b):
    # Align a to nearest higher multiple of b
    a = numpy.int32(a)
    b = numpy.int32(b)
    return (a - a % b + b) if (a % b != 0) else a

def iAlignDown(a, b):
    # Align a to nearest lower multiple of b
    a = numpy.int32(a)
    b = numpy.int32(b)
    return a - a % b

#How do I make my own kernel. The shape of this one is 1D. Look into how to convolution_cuda is implemented
def gaussian_kernel(width = KERNEL_W, sigma = 4.0):
    assert width == numpy.floor(width),  'argument width should be an integer!'
    radius = (width - 1)/2.0
    x = numpy.linspace(-radius,  radius,  width)
    x = numpy.float32(x)
    sigma = numpy.float32(sigma)
    filterx = x*x / (2 * sigma * sigma)
    filterx = numpy.exp(-1 * filterx)
    assert filterx.sum()>0,  'something very wrong if gaussian kernel sums to zero!'
    filterx /= filterx.sum()
    return filterx

#must set radius to 1 for this to work
def sobel_kernel_horiz(width = 3):
    filterx = numpy.array([1,0,-1])
    filterx = numpy.float32(filterx)
    return filterx

def sobel_kernel_vert(width = 3):
    filtery = numpy.array([1,2,1])
    filtery = numpy.float32(filtery)
    return filtery

#this only works with width = 3
def smooth_kernel(width = 3):
    filterx = numpy.array([1]*width)*(1/3)
    #filterx = numpy.array([1,2,1])*(1/4)
    filterx = numpy.float32(filterx)
    return filterx



def derivative_of_gaussian_kernel(width = KERNEL_W, sigma = 4):
    assert width == numpy.floor(width),  'argument width should be an integer!'
    radius = (width - 1)/2.0
    x = numpy.linspace(-radius,  radius,  width)
    x = numpy.float32(x)
    # The derivative of a gaussian is really just a gaussian times x, up to scale.
    filterx = gaussian_kernel(width,  sigma)
    filterx *= x
    # Rescale so that filter returns derivative of 1 when applied to x:
    scale = (x * filterx).sum()
    filterx /= scale
    # Careful with sign; this will be uses as a ~convolution kernel, so should start positive, then go negative.
    filterx *= -1.0
    return filterx

def test_derivative_of_gaussian_kernel():
    width = 20
    sigma = 10.0
    filterx = derivative_of_gaussian_kernel(width,  sigma)
    x = 2 * numpy.arange(0, width)
    x = numpy.float32(x)
    response = (filter * x).sum()
    assert abs(response - (-2.0)) < .0001, 'derivative of gaussian failed scale test!'
    width = 19
    sigma = 10.0
    filterx = derivative_of_gaussian_kernel(width,  sigma)
    x = 2 * numpy.arange(0, width)
    x = numpy.float32(x)
    response = (filterx * x).sum()
    assert abs(response - (-2.0)) < .0001, 'derivative of gaussian failed scale test!'

def convolution_cuda(sourceImage,  filterx,  filtery):
    # Perform separable convolution on sourceImage using CUDA.
    # Operates on floating point images with row-major storage.
    destImage = sourceImage.copy()
    assert sourceImage.dtype == 'float32',  'source image must be float32'
    (imageHeight,  imageWidth) = sourceImage.shape
    assert filterx.shape == filtery.shape == (KERNEL_W, ) ,  'Kernel is compiled for a different kernel size! Try changing KERNEL_W'
    filterx = numpy.float32(filterx)
    filtery = numpy.float32(filtery)
    DATA_W = iAlignUp(imageWidth, 16);
    DATA_H = imageHeight;
    BYTES_PER_WORD = 4;  # 4 for float32
    DATA_SIZE = DATA_W * DATA_H * BYTES_PER_WORD;
    KERNEL_SIZE = KERNEL_W * BYTES_PER_WORD;
    # Prepare device arrays
    destImage_gpu = cuda.mem_alloc_like(destImage)
    sourceImage_gpu = cuda.mem_alloc_like(sourceImage)
    intermediateImage_gpu = cuda.mem_alloc_like(sourceImage)
    cuda.memcpy_htod(sourceImage_gpu, sourceImage)
    cuda.memcpy_htod(d_Kernel_rows,  filterx) # The kernel goes into constant memory via a symbol defined in the kernel
    cuda.memcpy_htod(d_Kernel_columns,  filtery)
    # Call the kernels for convolution in each direction.
    blockGridRows = (iDivUp(DATA_W, ROW_TILE_W), DATA_H)
    blockGridColumns = (iDivUp(DATA_W, COLUMN_TILE_W), iDivUp(DATA_H, COLUMN_TILE_H))
    threadBlockRows = (KERNEL_RADIUS_ALIGNED + ROW_TILE_W + KERNEL_RADIUS, 1, 1)
    threadBlockColumns = (COLUMN_TILE_W, 8, 1)
    DATA_H = numpy.int32(DATA_H)
    DATA_W = numpy.int32(DATA_W)
    grid_rows = tuple([int(e) for e in blockGridRows])
    block_rows = tuple([int(e) for e in threadBlockRows])
    grid_cols = tuple([int(e) for e in blockGridColumns])
    block_cols = tuple([int(e) for e in threadBlockColumns])
    #TESTING CODE
   # print("Block rows \n",block_rows)
  #  print("BLock columns \n",block_cols)
    convolutionRowGPU(intermediateImage_gpu,  sourceImage_gpu,  DATA_W,  DATA_H,  grid=grid_rows,  block=block_rows)
    convolutionColumnGPU(destImage_gpu,  intermediateImage_gpu,  DATA_W,  DATA_H,  numpy.int32(COLUMN_TILE_W * threadBlockColumns[1]),  numpy.int32(DATA_W * threadBlockColumns[1]),  grid=grid_cols,  block=block_cols)

    # Pull the data back from the GPU.
    cuda.memcpy_dtoh(destImage,  destImage_gpu)
    return destImage

#m is our input bfKernel, which we base our approximation off
#This implemetnation is probably garabge.
def low_rank_approx(m,image,rank = 1):
  U,E,V = numpy.linalg.svd(m)
  #mn = numpy.zeros_like(m)
  #score = 0.0
  
  data = numpy.array(image)
  redChan = data[:,:,0]
  greenChan = data[:,:, 1]
  blueChan = data[:, : , 2]

  original = redChan #numpy.random.rand(768,  1024) * 255
  original = numpy.float32(original)

  original2 = greenChan
  original2 = numpy.float32(original2)

  original3 = blueChan
  original3 = numpy.float32(original3)
  
  destImage = original.copy()
  destImage2 = original2.copy()
  destImage3 = original3.copy()
  #destImage[:] = numpy.nan
  #destImage2[:] = numpy.nan#I think it's just emptied
  #destImage3[:] = numpy.nan#I think it's just emptied

#this is just a guess as to how it works, actually do the research~!!1
#THis is probably totally wrong, Give up for now and find out what actually works
  print(E)
  UPart = U[:,0]*-1
  VPart = V[0,:]*-1
  UPart = numpy.float64(UPart * numpy.sqrt(E[0]))
  VPart = numpy.float64(VPart * numpy.sqrt(E[0]))
  filtery = UPart
  filterx = VPart
  destImage = convolution_cuda(destImage,  filtery,  filterx)#*E[i]
    
  destImage2 = convolution_cuda(destImage2,  filtery,  filterx)#*E[i]
    
  destImage3 = convolution_cuda(destImage3,  filtery,  filterx)#*E[i]
  dataConcatenated = numpy.dstack((destImage, destImage2, destImage3)).astype(numpy.uint8)#depth stack!!
  imageFinal = Image.fromarray(dataConcatenated) 
  return imageFinal
"""
  for i in range(rank):
    filtery = U[:, i]
    filterx = V[i, :]
    
    destImage = convolution_cuda(destImage,  filterx,  filtery)#*E[i]
    
    destImage2 = convolution_cuda(destImage2,  filterx,  filtery)#*E[i]
    
    destImage3 = convolution_cuda(destImage3,  filterx,  filtery)#*E[i]
    
    #np.outer represents our convoluto
    #mn += E[i] * np.outer(U[:,i], V[i,:])
    score += E[i]
    "
  dataConcatenated = numpy.dstack((destImage, destImage2, destImage3)).astype(numpy.uint8)#depth stack!!
  imageFinal = Image.fromarray(dataConcatenated)

  print('Approximation percentage, ', score / numpy.sum(E))
  return imageFinal
"""
  #return mn
  
"""
Kernel is a numpy 2D matrix, which will be separated using SVD
image is also a numpy 2D matrix
rank determines number of iterations. Comes at a performance cost.

If this works with a bokeh effect kernel and guassian, you're golden, ready to integrte.
"""
def low_rank_approx_single_channel(kernel,image,rank = 1):
  U,E,V = numpy.linalg.svd(kernel/numpy.sum(kernel))#normalized kernel
  newImg = numpy.float32(numpy.zeros_like(image))#For rgb, this would be a 3D matrix

  channel = numpy.float32(image)
  for i in range(0, rank):
      UPart = U[:,i]#*-1
      VPart = V[i,:]#*-1
      UPart = UPart# * numpy.sqrt(E[i])
      VPart = VPart# * numpy.sqrt(E[i])
      filtery = UPart
      filterx = VPart
      newImg += convolution_cuda(channel,  filtery,  filterx)*E[i]
  return newImg

"""
NOTE: Even after integration, must confirm this works.
"""

def low_rank_approx_rgb_channel(kernel,image,rank = 1):

  U,E,V = numpy.linalg.svd(kernel/numpy.sum(kernel))#normalized kernel
  newImg = numpy.float32(numpy.zeros_like(image))#For rgb, this would be a 3D matrix

  for c in range(0,3):
      channel = numpy.float32(image[:,:,c])
      for i in range(0, rank):
        UPart = U[:,i]#*-1
        VPart = V[i,:]#*-1
        UPart = UPart# * numpy.sqrt(E[i])
        VPart = VPart# * numpy.sqrt(E[i])
        filtery = UPart
        filterx = VPart
        newImg[:,:,c] += convolution_cuda(channel,  filtery,  filterx)*E[i]
  return newImg


def test_brighter_fatter():
    """
    bfKernel = numpy.float64(numpy.loadtxt('bfKernel.txt'))
    image = Image.open('lena.png')
    
    finalImage = low_rank_approx(bfKernel,image,rank = 4)
    finalImage.save("BrighterFatterImg.png")
    """
    bfKernel = numpy.float64(numpy.loadtxt('bfKernel.txt'))
    
    image = Image.open('fitstest.png')
    data = numpy.array(image)#[:,:,0]#gets only red layer
    finalImage = low_rank_approx_rgb_channel(bfKernel, data, rank = 4)#low_rank_approx_single_channel(bfKernel, data, rank = 4)
    finalImagePicture = Image.fromarray(finalImage.astype(numpy.uint8))#uint8 i think downgrades to 8 bit image...
    finalImagePicture.save("bfImage.png")

def approx_convolve(kernel, input_matrix):
    kernel = numpy.float64(kernel)#this might not be right for lsst codebase
    
def test_gauss_separable():
    gauss = gaussian_kernel()
    boxKernel = numpy.float64(numpy.outer(gauss,gauss))
    image = Image.open('lena.png')
    data = numpy.array(image)
    finalImage = low_rank_approx_rgb_channel(boxKernel, data, rank = 1)
    finalImagePicture = Image.fromarray(finalImage.astype(numpy.uint8))
    finalImagePicture.save("GaussImg.png")
    
def bokeh_test():
    x, y = numpy.meshgrid(numpy.linspace(-1, 1, 25), numpy.linspace(-1, 1, 25))
    difference_of_gaussians = numpy.exp(-5 * (x*x+y*y))-numpy.exp(-6 * (x*x+y*y))
    difference_of_gaussians /= numpy.max(difference_of_gaussians)
    circle = numpy.array(x*x+y*y < 0.8, dtype='float64')

    image = Image.open('lena.png')
    data = numpy.array(image)
    finalImage = low_rank_approx_rgb_channel(difference_of_gaussians, data, rank = 3)
    finalImagePicture = Image.fromarray(finalImage.astype(numpy.uint8))
    finalImagePicture.save("BokehImg.png")
    
    circleFinalImg = low_rank_approx_rgb_channel(circle, data, rank = 1)
    circleFinalPicture = Image.fromarray(circleFinalImg.astype(numpy.uint8))
    circleFinalPicture.save("CircleImgRank1.png")

def test_bad_bf():
    # Test the convolution kernel.
    # Generate or load a test image    

    image = Image.open('lena.png')
    data = numpy.array(image)
    redChan = data[:,:,0]
    greenChan = data[:,:, 1]
    blueChan = data[:, : , 2]
    
    original = redChan #numpy.random.rand(768,  1024) * 255
    original = numpy.float32(original)
    
    original2 = greenChan
    original2 = numpy.float32(original2)
    
    original3 = blueChan
    original3 = numpy.float32(original3)
    
        # You probably want to display the image using the tool of your choice here.
    filterx = numpy.loadtxt("rowBfKernel.txt")#smooth_kernel() #figure out how to make your own kernel 
    filtery = numpy.loadtxt("colBfKernel.txt")
    
    #To try out:
        #set filterx equal to some kernel thing manually (a numpy array)
        #print stuff
    
    destImage = original.copy()
    destImage[:] = numpy.nan
    destImage = convolution_cuda(original,  filterx,  filtery)
    
    destImage2 = original2.copy()
    destImage2[:] = numpy.nan#I think it's just emptied
    destImage2 = convolution_cuda(original2,  filterx,  filtery)
    
    destImage3 = original3.copy()
    destImage3[:] = numpy.nan#I think it's just emptied
    destImage3 = convolution_cuda(original3,  filterx,  filtery)
    
    # You probably want to display the result image using the tool of your choice here.
    dataConcatenated = numpy.dstack((destImage, destImage2, destImage3)).astype(numpy.uint8)#depth stack!!
    imageFinal = Image.fromarray(dataConcatenated)
    imageFinal.save("BadBfImg.png")

    print ('Done running the convolution kernel!')

def test_convolution_cuda():
    # Test the convolution kernel.
    # Generate or load a test image    

    image = Image.open('lena.png')
    data = numpy.array(image)
    redChan = data[:,:,0]
    greenChan = data[:,:, 1]
    blueChan = data[:, : , 2]
    
    original = redChan #numpy.random.rand(768,  1024) * 255
    original = numpy.float32(original)
    
    original2 = greenChan
    original2 = numpy.float32(original2)
    
    original3 = blueChan
    original3 = numpy.float32(original3)
    """
    #GAUSSIAN BLUR
    # You probably want to display the image using the tool of your choice here.
    filterx = gaussian_kernel()#figure out how to make your own kernel 
    #To try out:
        #set filterx equal to some kernel thing manually (a numpy array)
        #print stuff
    
    destImage = original.copy()
    destImage[:] = numpy.nan
    destImage = convolution_cuda(original,  filterx,  filterx)
    
    destImage2 = original2.copy()
    destImage2[:] = numpy.nan#I think it's just emptied
    destImage2 = convolution_cuda(original2,  filterx,  filterx)
    
    destImage3 = original2.copy()
    destImage3[:] = numpy.nan#I think it's just emptied
    destImage3 = convolution_cuda(original3,  filterx,  filterx)
    # You probably want to display the result image using the tool of your choice here.
    """
    
    """
    SOBEL (EDGE DETECTION)
        # You probably want to display the image using the tool of your choice here.
    filterx = sobel_kernel_horiz() #figure out how to make your own kernel 
    filtery = sobel_kernel_vert()
    #To try out:
        #set filterx equal to some kernel thing manually (a numpy array)
        #print stuff
    
    destImage = original.copy()
    destImage[:] = numpy.nan
    destImage = convolution_cuda(original,  filterx,  filtery)
    destImage = convolution_cuda(destImage, filtery, filterx)
    
    destImage2 = original2.copy()
    destImage2[:] = numpy.nan#I think it's just emptied
    destImage2 = convolution_cuda(original2,  filterx,  filtery)
    destImage2 = convolution_cuda(destImage2, filtery, filterx)
    
    destImage3 = original2.copy()
    destImage3[:] = numpy.nan#I think it's just emptied
    destImage3 = convolution_cuda(original3,  filterx,  filtery)
    destImage3 = convolution_cuda(destImage3, filtery, filterx)
    
    """
    
        # You probably want to display the image using the tool of your choice here.
    filterx = smooth_kernel() #figure out how to make your own kernel 
    #To try out:
        #set filterx equal to some kernel thing manually (a numpy array)
        #print stuff
    
    destImage = original.copy()
    destImage[:] = numpy.nan
    destImage = convolution_cuda(original,  filterx,  filterx)
    
    destImage2 = original2.copy()
    destImage2[:] = numpy.nan#I think it's just emptied
    destImage2 = convolution_cuda(original2,  filterx,  filterx)
    
    destImage3 = original3.copy()
    destImage3[:] = numpy.nan#I think it's just emptied
    destImage3 = convolution_cuda(original3,  filterx,  filterx)
    
    # You probably want to display the result image using the tool of your choice here.
    dataConcatenated = numpy.dstack((destImage, destImage2, destImage3)).astype(numpy.uint8)#depth stack!!
    imageFinal = Image.fromarray(dataConcatenated)
    imageFinal.save("FinalConvolvedImageSmooth.png")

    print ('Done running the convolution kernel!')

if __name__ == '__main__':
    #test_convolution_cuda()
    #bokeh_test()
    #test_bad_bf()
    
    start = cuda.Event()#time.time()
    end = cuda.Event()
    start.record()
    test_brighter_fatter()
    end.record()
    end.synchronize()
    secs = start.time_till(end)*1e-3
    print("Seconds: ", secs)
    #end = time.time()
    #print(end-start, " seconds")
    
    #test_gauss_separable()
    #test_derivative_of_gaussian_kernel()
    #boo = raw_input('Pausing so you can look at results... <Enter> to finish...')
