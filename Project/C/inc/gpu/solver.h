#ifndef _SOLVER_H_
#define _SOVLER_H_

#include <cuda.h>
#include <cufft.h>


cufftHandle rfft;
cufftHandle irfft;

dim3 grid, threads;

__global__ void cube(double* c, double* cube);
__global__ void first_order(cufftDoubleComplex *c_hat, cufftDoubleComplex* f_hat, double dt, double hh);
__global__ void second_order(cufftDoubleComplex *c_hat, cufftDoubleComplex* c_hat_prev, cufftDoubleComplex* f_hat, cufftDoubleComplex* f_hat_prev, double dt, double hh);

void step(double* c, double dt);
void init_solver(double *c);
void free_solver();
void cudaGetSolution(double *c);

#endif
