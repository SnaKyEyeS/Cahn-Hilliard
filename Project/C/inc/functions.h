#ifndef FUNCTIONS_H_INCLUDED
#define FUNCTIONS_H_INCLUDED

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fftw3.h>
#include "const.h"

#define REAL 0
#define CPLX 1

void RungeKutta4(double* c, double dt);
void f(double* c, double* dc);
void laplacian(double* c, double h, double* delsq);

void free_functions(void);

// Temp variables for RungeKutta4 function
static double* k1  = (double*) malloc(N*N*sizeof(double));
static double* k2  = (double*) malloc(N*N*sizeof(double));
static double* k3  = (double*) malloc(N*N*sizeof(double));
static double* k4  = (double*) malloc(N*N*sizeof(double));
static double* tmp = (double*) malloc(N*N*sizeof(double));

// Temp variables for f function
static double* delsq = (double*) malloc(N*N*sizeof(double));

// FFT input & output variables (need to be initialised at the start)
static fftw_complex *cval = fftw_alloc_complex(N*(N/2+1));
static double       *rval = fftw_alloc_real(N*N);

// FFTW plans for real-valued forward & backard 2-D DFT
static const fftw_plan rfft2  = fftw_plan_dft_r2c_2d(N, N, rval, cval, FFTW_EXHAUSTIVE);
static const fftw_plan irfft2 = fftw_plan_dft_c2r_2d(N, N, cval, rval, FFTW_EXHAUSTIVE);

// TODO: it may be best to define the FFTW plans in the main function
// so we can properly destroy them, etc.
// Though they'd have to be passed in argument or whatnot - TBD.
// See also new-array execute functions:
// http://www.fftw.org/doc/New_002darray-Execute-Functions.html#New_002darray-Execute-Functions

#endif
