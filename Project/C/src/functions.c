#include "functions.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

double* RungeKutta4(double* C, double dt, double a){

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

  double* delsq = (double*)malloc(N*N*sizeof(double)); // si tu trouves un meilleur nom que array hésites pas haha

  laplacian(C, 1.0/(N-1), delsq);
  for(int i=0; i<N*N; i++){
    C[i] = pow(C[i],3) - C[i] - a*a*delsq[i];
  }
  laplacian(C, 1.0/(N-1), delsq);

  return delsq;

}

void laplacian(double* u, double h, double* delsq){

    fftw_complex spectrum[N*(N/2+1)];
    fftw_plan thePlan;

    // Forward 2D real-valued FFT
    thePlan = fftw_plan_dft_r2c_2d(N, N, u, spectrum, FFTW_ESTIMATE);
    fftw_execute(thePlan);

    // Take the derivative
    int l, ind;
    double k;
    double factor = .25*h*h / (M_PI*M_PI);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N/2+1; j++) {

            // Wavenumber
            l = (i < N/2) ? i : i-N;
            k = -factor * (j*j + l*l);

            // Multiply by (ik)²
            ind = i*(N/2+1)+j;
            spectrum[ind][REAL] = k*spectrum[ind][REAL];
            spectrum[ind][CPLX] = k*spectrum[ind][CPLX];
        }
    }

    // Backward 2D real-valued FFT
    thePlan = fftw_plan_dft_c2r_2d(N, N, spectrum, delsq, FFTW_ESTIMATE);
    fftw_execute(thePlan);

    // Free memory
    fftw_destroy_plan(thePlan);
}
