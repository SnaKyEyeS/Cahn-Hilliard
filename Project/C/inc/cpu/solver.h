#ifndef _SOLVER_H_
#define _SOVLER_H_

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>


void step(double *c, double dt);
void init_solver(double *c);
void free_solver();

// FFTW plans for real-valued forward & backard 2-D DFT
fftw_complex *cval;
double       *rval;
fftw_plan rfft2;
fftw_plan irfft2;

#endif
