#ifndef _SOLVER_H_
#define _SOVLER_H_

#include <cuda.h>
#include <cufft.h>


typedef cufftDoubleComplex complex;

cufftHandle rfft;
cufftHandle irfft;

dim3 grid, threads;

__global__ void cube(double* c, double* cube);
__global__ void first_order(complex *c_hat, complex* f_hat, double dt, double hh, complex *out);
__global__ void second_order(complex *c_hat, complex* c_hat_prev, complex* f_hat, complex* f_hat_prev, double dt, double hh, complex *out);

void step(double* c, double dt);
void init_solver(double *c);
void free_solver();
void cudaGetSolution(double *c);

#endif
