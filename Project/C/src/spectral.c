#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "matplotlibcpp.h"
#include "functions.h"

namespace plt = matplotlibcpp;

int main(int argc, char *argv[]){
  // parameters
  double a = 0.01;
  int n = 128;
  double dt = pow(10,-6)/4.0;
  int n_step = 12000*4;
  int skip_frame = 10;
  char title[50];

  //init X and h
  double X_max = 1.0;
  double step = X_max/double(n-1);

  double* X = (double*)calloc(n, sizeof(double));
  double* h = (double*)calloc(n, sizeof(double));

  for(int i=0; i<n; i++){
    X[i] = i*step;
    h[i] = step;
  }


  //init C
  double* C = (double*)calloc(n*n, sizeof(double));

  float* plot_C = (float*)calloc(n*n, sizeof(float));

  for(int i=0; i<n*n; i++){
    C[i] = 2.0*((double)rand() / (double)RAND_MAX ) - 1.0;
  }

  //loop on time:
  for(int t=0; t<n_step; t++){

    C = RungeKutta4(C, dt, a);

    if(t%skip_frame == 0){

      for(int i=0; i<n*n; i++){
        plot_C[i] = float(C[i]);
      }

      plt::clf();

  		sprintf(title, "Time = %f", t*dt*skip_frame);
  		const int colors = 1;

      plt::title(title);
      plt::imshow(&(plot_C[0]), 128, 128, colors);

      // Show plots
      plt::pause(0.1);
    }
  }
}
