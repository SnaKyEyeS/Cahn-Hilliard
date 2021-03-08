#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <time.h>
#include "functions.h"
#include "window.h"
#include "shaders.h"


int main(int argc, char* argv[]) {

    // Initialise window
    GLFWwindow *window = init_window();

    // Initialise shaders
    init_shaders();

    // Create Vertex Array Object
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Create a Vertex Buffer Object for positions
    GLuint vbo_pos;
    glGenBuffers(1, &vbo_pos);

    GLfloat positions[2*N*N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int ind = i*N+j;
            positions[2*ind  ] = (float)(-1.0 + 2.0*i/(N-1));
            positions[2*ind+1] = (float)(-1.0 + 2.0*j/(N-1));
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);

    // Specify vbo_pos' layout
    GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);

    // Create an Element Buffer Object and copy the element data to it
    GLuint ebo;
    glGenBuffers(1, &ebo);

    GLuint elements[4*(N-1)*(N-1)];
    for (int i = 0; i < N-1; i++) {
        for (int j = 0; j < N-1; j++) {
            int ind  = i*N+j;
            int ind_ = i*(N-1)+j;

            elements[4*ind_  ] = ind;
            elements[4*ind_+1] = ind+1;
            elements[4*ind_+2] = ind+N;
            elements[4*ind_+3] = ind+N+1;
        }
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(elements), elements, GL_STATIC_DRAW);

    // Simulation parameters
    int n = N;
    int t = 0;
    double dt = 1e-6/4;
    double skip = 10;

    // Initialise Cahn-Hilliard solver
    init_functions();
    double *c = (double*) malloc(n*n*sizeof(double));
    for (int i = 0; i < n*n; i++) {
        c[i] = 2.0*((double)rand() / (double)RAND_MAX ) - 1.0;
    }

    // Create a Vertex Buffer Object for colors
    GLuint vbo_colors;
    glGenBuffers(1, &vbo_colors);

    GLfloat colors[N*N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int ind = i*N+j;
            colors[ind] = (float) ((c[ind] + 1.0)/2.0);
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STREAM_DRAW);

    // Specify vbo_color's layout
    GLint colAttrib = glGetAttribLocation(shaderProgram, "color");
    glEnableVertexAttribArray(colAttrib);
    glVertexAttribPointer(colAttrib, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);


    clock_t begin, end;
    while (!glfwWindowShouldClose(window)) {
        // Timestepping
        begin = clock();
        for (int i = 0; i < skip; i++) {
            RungeKutta4(c, dt);
            t++;
        }
        end = clock();

        // Event input
        glfwPollEvents();
        if (drag) {
            double xpos, ypos;
            glfwGetCursorPos(window, &xpos, &ypos);
            // printf("(%f, %f)\n", xpos, ypos);
        }
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, GL_TRUE);
        }

        // Update graphics
        // begin = clock();
        glfwSwapBuffers(window);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Update plot
        for (int i = 0; i < N*N; i++) {
            colors[i] = (float) ((c[i] + 1.0)/2.0);
        }
        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
        glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STREAM_DRAW);

        // Draw elements
        glDrawElements(GL_LINES_ADJACENCY, 4*(N-1)*(N-1), GL_UNSIGNED_INT, 0);
        // end = clock();

        // Print stuff
        printf("Time = %f\n", (double)(end-begin)/CLOCKS_PER_SEC);
        // printf("\rIter = %5d, Time = %.6f  ", t, t*dt);
        // fflush(stdout);
    }

    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &vbo_pos);
    glDeleteBuffers(1, &vbo_colors);

    glDeleteVertexArrays(1, &vao);

    glfwDestroyWindow(window);
    glfwTerminate();
    free_functions();
    free_shaders();
    free(c);

    return EXIT_SUCCESS;
}
