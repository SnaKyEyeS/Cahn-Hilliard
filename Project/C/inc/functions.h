#ifndef FUNCTIONS_H_INCLUDED
#define FUNCTIONS_H_INCLUDED

#include <fftw3.h>
#include "const.h"

#define REAL 0
#define CPLX 1

double* RungeKutta4(double* C, double dt, double a);

double* f(double* C, double a);

void laplacian(double* u, double h, double* delsq);

#endif
