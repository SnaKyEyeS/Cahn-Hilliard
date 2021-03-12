#ifndef _KERNEL_H_
#define _KERNEL_H_

#include <cuda.h>
#include <cufft.h>
#include "const.h"



cufftDoubleComplex *cval;

cufftHandle rfft;
cufftHandle irfft;

dim3 grid, threads;


__global__ void deriv(double h, cufftDoubleComplex* cval);

__global__ void k12_sum(double* c, double* k, double* tmp, double dt);

__global__ void k3_sum(double* c, double* k, double* tmp, double dt);

__global__ void k_sum_tot(double* c, double* k1, double* k2, double* k3, double* k4, double dt);

__global__ void inside_deriv(double* c, double* delsq);


void cufft_laplacian(double* c, double h, double* delsq);
void RungeKutta4(double* c, double dt);
void f(double* c, double* dc);
void laplacian(double* c, double h, double* delsq);

// Temp variables for RungeKutta4 function
double* k1;
double* k2;
double* k3;
double* k4;
double* tmp;
double* delsq;


void init_cuda(double* c_gpu);
void free_cuda(double* c_gpu);

void copy_cuda_H2D(double* c_gpu, double* c);
void copy_cuda_D2H(double* c, double* c_gpu);


#endif
