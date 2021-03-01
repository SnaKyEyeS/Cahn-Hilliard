#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "matplotlibcpp.h"
#include "functions.h"
#include "const.h"

namespace plt = matplotlibcpp;

int main(int argc, char *argv[]){
  // parameters
  double a = 0.01;
  double dt = pow(10,-6)/4.0;
  int n_step = 12000*4;
  int skip_frame = 10;
  char title[50];

  //init X and h
  double X_max = 1.0;
  double step = X_max/double(N-1);

  double* X = (double*)calloc(N, sizeof(double));
  double* h = (double*)calloc(N, sizeof(double));

  for(int i=0; i<N; i++){
    X[i] = i*step;
    h[i] = step;
  }


  //init C
  double* C = (double*)calloc(N*N, sizeof(double));

  float* plot_C = (float*)calloc(N*N, sizeof(float));

  for(int i=0; i<N*N; i++){
    C[i] = 2.0*((double)rand() / (double)RAND_MAX ) - 1.0;
  }


  //loop on time:
  for(int t=0; t<n_step; t++){

    RungeKutta4(C, dt);

    if(t%skip_frame == 0){

      for(int i=0; i<N*N; i++){
        plot_C[i] = float(C[i]);
      }

      plt::clf();

  		sprintf(title, "Time = %f", t*dt);
  		const int colors = 1;

      plt::title(title);
      plt::imshow(&(plot_C[0]), N, N, colors);

      // Show plots
      plt::pause(1e-10);
    }
  }

  free_functions();
  return 1;
}
