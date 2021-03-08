#ifndef FUNCTIONS_H_INCLUDED
#define FUNCTIONS_H_INCLUDED

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fftw3.h>
#include "const.h"

#ifdef USE_OPENMP
#include <omp.h>
#endif

#define REAL 0
#define CPLX 1

void RungeKutta4(double* c, double dt);
void f(double* c, double* dc);
void laplacian(double* c, double h, double* delsq);

void init_functions(void);
void free_functions(void);

// Temp variables for RungeKutta4 function
double* k1;
double* k2;
double* k3;
double* k4;
double* tmp;

// Temp variables for f function
double* delsq;

// FFT input & output variables (need to be initialised at the start)
fftw_complex *cval;
double       *rval;

// FFTW plans for real-valued forward & backard 2-D DFT
fftw_plan rfft2;
fftw_plan irfft2;

// TODO: it may be best to define the FFTW plans in the main function
// so we can properly destroy them, etc.
// Though they'd have to be passed in argument or whatnot - TBD.
// See also new-array execute functions:
// http://www.fftw.org/doc/New_002darray-Execute-Functions.html#New_002darray-Execute-Functions

#endif
