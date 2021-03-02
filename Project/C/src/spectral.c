#include "BOV.h"
#include <time.h>
#include <math.h>
// #include "functions.h"

int main(int argc, char* argv[]) {

    // Size of simulation
    int n = 128;

    // Create window
    bov_window_t* window = bov_window_new(n, n, "LMECA2300");
    window->param.zoom = 2.0/n;
    window->param.translate[0] = -n/2.0;
    window->param.translate[1] = -n/2.0;
    bov_window_set_color(window, (GLfloat[4]) {0.3, 0.3, 0.3, 1});

    // Init C
    double *c = (double*) malloc(n*n*sizeof(double));
    for (int i = 0; i < n*n; i++) {
        c[i] = 2.0*((double)rand() / (double)RAND_MAX ) - 1.0;
    }

    // Draw initial condition
    float none[4] = {0.0,0.0,0.0,1};
    float color;
    clock_t begin, end;
    GLfloat data[5][3];

    do {
        begin = clock();

        // Update
        bov_window_update(window);

        // Draw points
        for (float i = 0.0; i < n; i++) {
            for (float j = 0.0; j < n; j++) {

                int ind = i*n+j;
                color = (float) (c[ind]+1.0)/2.0;

                data[0][0] = i;
                data[0][1] = j;
                data[0][2] = color;

                data[1][0] = i+1;
                data[1][1] = j;
                data[1][2] = color;

                data[2][0] = i+1;
                data[2][1] = j+1;
                data[2][2] = color;

                data[3][0] = i;
                data[3][1] = j+1;
                data[3][2] = color;

                data[4][0] = i;
                data[4][1] = j;
                data[4][2] = color;

                bov_points_t* points = bov_points_new_with_value(data, 5, GL_DYNAMIC_DRAW);
                bov_points_set_color(points, none);
                bov_points_set_width(points, 0);
                bov_points_set_outline_color(points, none);
                bov_points_set_outline_width(points, 0);
                bov_triangles_draw(window, points, 0, 3);
                bov_triangles_draw(window, points, 2, 3);

                bov_points_delete(points);
            }
        }

        end = clock();
        printf("Time = %f\n", (double)(end-begin)/CLOCKS_PER_SEC);

    } while(!bov_window_should_close(window));


    bov_window_delete(window);

    return EXIT_SUCCESS;
}
