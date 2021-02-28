#include "functions.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

double* RungeKutta4(double* C, double dt, double a){

  int N = sizeof(C)/sizeof(C[0]);

  double* Cnext = (double*)calloc(N*N, sizeof(double));

  double* K1 = (double*)calloc(N*N,sizeof(double));
  double* K2 = (double*)calloc(N*N,sizeof(double));
  double* K3 = (double*)calloc(N*N,sizeof(double));
  double* K4 = (double*)calloc(N*N,sizeof(double));

  K1 = f(C, a);
  for(int i=0; i<N*N; i++){
      Cnext[i] = C[i] + (1.0/2.0)*K1[i];
  }

  K2 = f(Cnext, a);
  for(int i=0; i<N*N; i++){
      Cnext[i] = C[i] + (1.0/2.0)*K2[i];
  }

  K3 = f(Cnext, a);
  for(int i=0; i<N*N; i++){
      Cnext[i] = C[i] + K3[i];
  }

  K4 = f(Cnext, a);

  for(int i=0; i<N*N; i++){
      Cnext[i] = C[i] + (1.0/6.0)*(K1[i]+2*K2[i]+2*K3[i]+K4[i]);
  }
  return Cnext;
}

double* f(double* C, double a){
  int N = sizeof(C)/sizeof(C[0]);

  double* array = (double*)calloc(N*N, sizeof(double)); // si tu trouves un meilleur nom que array hésites pas haha

  array = laplacian(C);

  for(int i=0; i<N*N; i++){
    array[i] = pow(C[i],3) - C[i] - a*a*array[i];
  }

  return laplacian(array);

}

double* laplacian(double* C){
  return C;
}
