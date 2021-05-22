#ifndef _SOLVER_H_
#define _SOVLER_H_

#include <math.h>
#include <complex.h>
#include <stdlib.h>
#include <string.h>
#include <fftw3.h>

typedef enum {
    IMEX,
    ETDRK4,
} SOLVER_TYPE;


void   step(double dt);
void   imex(double dt);
void etdrk4(double dt);
void getSolution(double *c);
void non_linear_term(fftw_complex *c, fftw_complex *out);

void init_solver(double *c, double dt);
void free_solver();

// FFTW plans for real-valued forward & backard 2-D DFT
fftw_complex *cval;
double       *rval;
fftw_plan rfft2;
fftw_plan irfft2;

#endif
