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

void   imex(double dt);
__global__ void imex_bdf1(complex *c_hat_0, complex* f_hat, double dt, double hh, complex* c_hat);
__global__ void imex_bdf2(complex *c_hat_0, complex* c_hat_1, complex* f_hat_0, complex* f_hat_1, double dt, double hh, complex *c_hat);

void etdrk4(double dt);
__global__ void compute_fa(complex *c_hat, complex *Nu, double *e2, double *q, complex *fa);
__global__ void compute_fb(complex *c_hat, complex *Na, double *e2, double *q, complex *fb);
__global__ void compute_fc(complex *c_hat, complex *Nu, complex *Nb, double *e2, double *q, complex *fc);
__global__ void etdrk4_next(complex* c_hat, complex *Nu, complex *Na, complex *Nb, complex *Nc, double *e1, double *f1, double *f2, double* f3);

void non_linear_term(complex *c_hat, complex *f_hat);
__global__ void scale(complex *c_hat, complex *out, double hh);
__global__ void f(double *c, double *f);
__global__ void deriv(complex *c_hat);

void step(double dt);
void init_solver(double *c, double dt);
void free_solver();
void getSolution(double *c);

#endif
