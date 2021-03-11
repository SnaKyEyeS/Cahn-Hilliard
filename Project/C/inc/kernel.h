#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <cuda.h>
#include <cufft.h>
#include "const.h"


float* delsq_gpu;
float* rval_gpu;
cufftComplex *cval_gpu;

cufftHandle rfft;
cufftHandle irfft;

dim3 grid, threads;


__global__ void deriv(float h, cufftComplex* cval);


void cufft_laplacian(float* c, float h, float* delsq);


void init_cuda(void);
void free_cuda(void);


#endif
