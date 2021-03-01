#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "matplotlibcpp.h"
#include "functions.h"
#include "const.h"

namespace plt = matplotlibcpp;


int main(int argc, char *argv[]){

    // Initialise value
    int ind;
    double h = 2*M_PI/N;
    double *f = (double *)malloc(N*N*sizeof(double));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            ind = i*N + j;
            f[ind] = cos(i*h)*sin(j*h);
        }
    }

    // Spectral derivative
    double *df = (double *)malloc(N*N*sizeof(double));
    laplacian(f, h, df);

    // Construct x, y, z
    std::vector<std::vector<double>> x, y, z;
    for (int i = 0; i < N; i++) {
        std::vector<double> x_row, y_row, z_row;
        for (int j = 0; j < N; j++) {
            // X & Y coords
            x_row.push_back(i*h);
            y_row.push_back(j*h);

            // The stuff to plot
            ind = i*N + j;
            z_row.push_back(df[ind] + 2*cos(i*h)*sin(j*h));
        }
        x.push_back(x_row);
        y.push_back(y_row);
        z.push_back(z_row);
    }

    // Plot results
    plt::plot_surface(x, y, z);
    plt::show();
}
