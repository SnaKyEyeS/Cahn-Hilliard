#ifndef _SOLVER_H_
#define _SOVLER_H_

#include <cuda.h>
#include <cufft.h>


// Temp variables for RungeKutta4 function
double* k1;
double* k2;
double* k3;
double* k4;
double* tmp;

double* c_gpu;
double* c_cube;

cufftDoubleComplex *cval;
cufftDoubleComplex *c_hat;

cufftHandle rfft;
cufftHandle irfft;

dim3 grid, threads;

__global__ void k12_sum(double* c, double* k, double* tmp, double dt);
__global__ void k3_sum(double* c, double* k, double* tmp, double dt);
__global__ void k_sum_tot(double* c, double* k1, double* k2, double* k3, double* k4, double dt);
__global__ void cube(double* c, double* cube);
__global__ void deriv(cufftDoubleComplex *c_hat, cufftDoubleComplex* cval, double hh);

void RungeKutta4(double* c, double dt);
void f(double* c, double* dc);

void init_solver(double *c);
void free_solver();
void cudaGetSolution(double *c);

#endif
