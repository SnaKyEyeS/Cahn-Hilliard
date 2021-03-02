#include "BOV.h"
#include <time.h>
#include <math.h>
#include "functions.h"
#include "plot.h"

int main(int argc, char* argv[]) {

    // Simulation parameters
    int n = N;
    int t = 0;
    double dt = 1e-6/4;
    double skip = 10;

    // Create window
    bov_window_t* window = bov_window_new(500, 500, "LMECA2300");
    window->param.zoom = 2.0/n;
    window->param.translate[0] = -n/2.0;
    window->param.translate[1] = -n/2.0;
    bov_window_set_color(window, (GLfloat[4]) {0.3, 0.3, 0.3, 1});

    // Init
    init_functions();
    double *c = (double*) malloc(n*n*sizeof(double));
    for (int i = 0; i < n*n; i++) {
        c[i] = 2.0*((double)rand() / (double)RAND_MAX ) - 1.0;
    }

    // Graphical loop
    clock_t begin, end;
    do {
        // Update
        bov_window_update(window);

        // Draw points
        // begin = clock();
        imshow(window, c, n, n);
        // end = clock();
        // printf("Time = %f\n", (double)(end-begin)/CLOCKS_PER_SEC);

        // Timestepping
        for (int i = 0; i < skip; i++) {
            RungeKutta4(c, 1e-6/4);
            t++;
        }

        printf("Iter = %5d; Time = %.6f\n", t, t*dt);

    } while(!bov_window_should_close(window));


    bov_window_delete(window);
    free_functions();
    return EXIT_SUCCESS;
}
