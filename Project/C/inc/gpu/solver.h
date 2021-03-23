#ifndef _SOLVER_H_
#define _SOVLER_H_

#include <cuda.h>
#include <cufft.h>

typedef cufftDoubleComplex complex;
typedef enum {
    RK4,
    IMEX,
    ETDRK4,
} SOLVER_TYPE;


cufftHandle rfft;
cufftHandle irfft;

dim3 grid, threads;

__global__ void cube(double* c, double* cube);
__global__ void imex_bdf1(complex *c_hat, complex* f_hat, double dt, double hh, complex *out);
__global__ void imex_bdf2(complex *c_hat, complex* c_hat_prev, complex* f_hat, complex* f_hat_prev, double dt, double hh, complex *out);

void step(double dt);
void init_solver(double *c, double dt);
void free_solver();
void getSolution(double *c);

#endif
