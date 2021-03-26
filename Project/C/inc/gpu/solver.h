#ifndef _SOLVER_H_
#define _SOVLER_H_

#include <cuda.h>
#include <cufft.h>

typedef cufftDoubleComplex gpuComplex;
typedef enum {
    RK4,
    IMEX,
    ETDRK4,
} SOLVER_TYPE;


cufftHandle rfft;
cufftHandle irfft;

void   imex(double dt);
__global__ void imex_bdf1(gpuComplex *c_hat_0, gpuComplex* f_hat, double dt, double hh, gpuComplex* c_hat);
__global__ void imex_bdf2(gpuComplex *c_hat_0, gpuComplex* c_hat_1, gpuComplex* f_hat_0, gpuComplex* f_hat_1, double dt, double hh, gpuComplex *c_hat);

void etdrk4(double dt);
__global__ void compute_fa(gpuComplex *c_hat, gpuComplex *Nu, double *e2, double *q, gpuComplex *fa);
__global__ void compute_fb(gpuComplex *c_hat, gpuComplex *Na, double *e2, double *q, gpuComplex *fb);
__global__ void compute_fc(gpuComplex *c_hat, gpuComplex *Nu, gpuComplex *Nb, double *e2, double *q, gpuComplex *fc);
__global__ void etdrk4_next(gpuComplex* c_hat, gpuComplex *Nu, gpuComplex *Na, gpuComplex *Nb, gpuComplex *Nc, double *e1, double *f1, double *f2, double* f3);

void non_linear_term(gpuComplex *c_hat, gpuComplex *f_hat);
__global__ void scale(gpuComplex *c_hat, gpuComplex *out, double hh);
__global__ void f(double *c, double *f);
__global__ void deriv(gpuComplex *c_hat);

void step(double dt);
void init_solver(double *c, double dt);
void free_solver();
void getSolution(double *c);

#endif
