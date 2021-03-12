#ifndef _SOLVER_H_
#define _SOVLER_H_

#include <math.h>
#include <stdlib.h>
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

void init_solver(double *c);
void free_solver();

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

#endif
