#include "BOV.h"
#include <math.h>

int main(int argc, char* argv[]) {

    // Size of simulation
    int n = 128;

    // Create window
    bov_window_t* window = bov_window_new(n, n, "LMECA2300");
    bov_window_enable_help(window);
    bov_window_set_color(window, (GLfloat[4]) {0.3, 0.3, 0.3, 1});

    // Init C
    double *c = (double*) malloc(n*n*sizeof(double));
    for (int i = 0; i < n*n; i++)
        c[i] = 2.0*((double)rand() / (double)RAND_MAX ) - 1.0;

    // Random
    double max =  1.0;
    double min = -1.0;

    // Draw initial condition
    float red[4] = {1.0,0.0,0.0,1.0};

    do {
        // Update
        bov_window_update_and_wait_events(window);

        // Draw points
        for (int i = 0; i < n*n; i++) {
            for (int j = 0; j < n*n; j++) {

                int ind = i*n+j;
                GLfloat (*data)[3] = malloc(sizeof(data[0])*3);
                data[0][0] = (float) i;
                data[0][1] = (float) j;
                data[0][2] = (c[ind]+1.0)/2.0;
                data[1][0] = (float) (i+1);
                data[1][1] = (float) j;
                data[1][2] = (c[ind]+1.0)/2.0;
                data[2][0] = (float) i;
                data[2][1] = (float) (j+1);
                data[2][2] = (c[ind]+1.0)/2.0;

                bov_points_t* points = bov_points_new_with_value(data, 3, GL_DYNAMIC_DRAW);
                bov_points_set_color(points, red);
                bov_points_set_width(points,0);
                bov_points_set_outline_color(points, red);
                bov_points_set_outline_width(points, 0);
                bov_triangles_draw(window, points, 0, BOV_TILL_END);

                bov_points_delete(points);
                free(data);
            }
        }

    } while(!bov_window_should_close(window));


    bov_window_delete(window);

    return EXIT_SUCCESS;
}
