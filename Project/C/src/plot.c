#include "plot.h"


void imshow(bov_window_t *w, double *z, int n1, int n2) {

    // Init stuff
    float none[4] = {0.0,0.0,0.0,1};
    float color;
    GLfloat data[5][3];

    for (float i = 0.0; i < n1; i++) {
        for (float j = 0.0; j < n2; j++) {

            int ind = i*n2+j;
            color = (float) (z[ind]+1.0)/2.0;

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
            bov_triangles_draw(w, points, 0, 3);
            bov_triangles_draw(w, points, 2, 3);

            bov_points_delete(points);
        }
    }

}
