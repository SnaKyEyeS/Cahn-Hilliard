#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <cuda.h>
#include <cufft.h>
#include "const.h"


double* delsq_gpu;
double* rval_gpu;
cufftDoubleComplex *cval_gpu;

cufftHandle rfft;
cufftHandle irfft;

dim3 grid, threads;


__global__ void deriv(double h, cufftDoubleComplex* cval);


void cufft_laplacian(double* c, double h, double* delsq);


void init_cuda(void);
void free_cuda(void);


#endif
