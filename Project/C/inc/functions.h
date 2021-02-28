#ifndef FUNCTIONS_H_INCLUDED
#define FUNCTIONS_H_INCLUDED

double* RungeKutta4(double* C, double dt, double a);

double* f(double* C, double a);

double* laplacian(double* C);

#endif
