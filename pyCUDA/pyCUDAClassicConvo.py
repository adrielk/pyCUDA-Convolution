# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 23:05:43 2020

@author: thead
"""
from PIL import Image
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy

#From my observation, d_Result is the entire image array, and d_Data is the kernel.
#Also dataW and dataH are the kernel's width and height respectively.
mod = SourceModule("""
__global__ void convolutionGPU(
    float *d_Result,
    float *d_Data,
    int dataW,
    int dataH,
    int KERNEL_RADIUS
    )
{
    // Data cache: threadIdx.x , threadIdx.y
    __shared__ float data[4 + KERNEL_RADIUS * 2][4 + KERNEL_RADIUS * 2];
    
    // global mem address of this thread
    const int gLoc = threadIdx.x +
            (blockIdx.x*blockDim.x) +
            (threadIdx.y* dataW) +
            (blockIdx.y * blockDim.y) * dataW;

    // load cache (32x32 shared memory, 16x16 threads blocks)
    // each threads loads four values from global memory into shared mem
    // if in image area, get value in global mem, else 0
    int x, y; // image based coordinate

    // original image based coordinate
    const int x0 = threadIdx.x + (blockIdx.x * blockDim.x);
    const int y0 = threadIdx.y + (blockIdx.y * blockDim.y);

    // case1: upper left
    x = x0 - KERNEL_RADIUS;
    y = y0 - KERNEL_RADIUS;
    if ( x < 0 || y < 0 )
        data[threadIdx.x][threadIdx.y] = 0;
    else
        data[threadIdx.x][threadIdx.y] = d_Data[ gLoc - KERNEL_RADIUS - (dataW * KERNEL_RADIUS)];

    // case2: upper right
    x = x0 + KERNEL_RADIUS;
    y = y0 - KERNEL_RADIUS;
    if ( x > dataW-1 || y < 0 )
        data[threadIdx.x + blockDim.x][threadIdx.y] = 0;
    else
        data[threadIdx.x + blockDim.x][threadIdx.y] = d_Data[gLoc + KERNEL_RADIUS - (dataW * KERNEL_RADIUS)];

    // case3: lower left
    x = x0 - KERNEL_RADIUS;
    y = y0 + KERNEL_RADIUS;
    if (x < 0 || y > dataH-1)
        data[threadIdx.x][threadIdx.y + blockDim.y] = 0;
    else
        data[threadIdx.x][threadIdx.y + blockDim.y] = d_Data[gLoc - KERNEL_RADIUS + (dataW * KERNEL_RADIUS)];

    // case4: lower right
    x = x0 + KERNEL_RADIUS;
    y = y0 + KERNEL_RADIUS;
    if ( x > dataW-1 || y > dataH-1)
        data[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = 0;
    else
        data[threadIdx.x + blockDim.x][threadIdx.y + blockDim.y] = d_Data[gLoc + KERNEL_RADIUS + (dataW * KERNEL_RADIUS)];

    __syncthreads();

    // convolution
    float sum = 0;
    x = KERNEL_RADIUS + threadIdx.x;
    y = KERNEL_RADIUS + threadIdx.y;
    for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++)
        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            sum += data[x + i][y + j] * d_Data[KERNEL_RADIUS + j] * d_Data[KERNEL_RADIUS + i];

    d_Result[gLoc] = sum;
}
  """)
  
func = mod.get_function("convolutionGPU")


image = Image.open('lena.png')
data = numpy.array(image)
redChan = data[:,:,0]
greenChan = data[:,:, 1]
blueChan = data[:, : , 2]

kernel = numpy.full((17,17), 1)
kernel = numpy.float32(kernel)

original = redChan #numpy.random.rand(768,  1024) * 255
original = numpy.float32(original)

# original2 = greenChan
# original2 = numpy.float32(original2)

# original3 = blueChan
# original3 = numpy.float32(original3)

image_gpu = cuda.mem_alloc(original.nbytes)
kernel_gpu = cuda.mem_alloc(kernel.nbytes)

cuda.memcpy_htod(image_gpu, original)
cuda.memcpy_htod(kernel_gpu, kernel)

func(image_gpu, kernel_gpu, 17, 17, 8 , block = (original.size[0], original.size[1], 1))

final_image = numpy.empty_like(original)
cuda.memcpy_dtoh(final_image, image_gpu)

imageOriginal = Image.fromarray(original)
imageFinal = Image.fromarray(final_image)
imageOriginal.save("ClassicConvolvedOriginal.png")
imageFinal.save("ClassicConvolved.png")


print ('Done running the convolution kernel!')







