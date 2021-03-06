#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdio.h>
#include <time.h>
#include "functions.h"
#include "window.h"


// Shader sources
const GLchar* vertexSource = R"glsl(
    #version 460 core
    in vec2 position;
    in float color;
    out float Color;
    void main() {
        Color = color;
        gl_Position = vec4(position, 0.0, 1.0);
    }
)glsl";
const GLchar* fragmentSource = R"glsl(
    #version 460 core
    in float Color;
    in vec2 position;
    out vec4 outColor;

    vec4 turbo(float x) {
        const vec4 kRedVec4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
        const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
        const vec4 kBlueVec4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
        const vec2 kRedVec2 = vec2(-152.94239396, 59.28637943);
        const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
        const vec2 kBlueVec2 = vec2(-89.90310912, 27.34824973);

        vec4 v4 = vec4( 1.0, x, x*x, x*x*x);
        vec2 v2 = v4.zw * v4.z;

        return vec4(
            dot(v4, kRedVec4)   + dot(v2, kRedVec2),
            dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
            dot(v4, kBlueVec4)  + dot(v2, kBlueVec2),
            1.0
        );
    }

    void main() {
        outColor = turbo(Color);
    }
)glsl";


int main(int argc, char* argv[]) {

    // Initialise window
    GLFWwindow *window = init_window();

    // Create Vertex Array Object
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // Create and compile the vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    // Create and compile the fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);

    // Link the vertex and fragment shader into a shader program
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glBindFragDataLocation(shaderProgram, 0, "outColor");
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);

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

    GLuint elements[6*(N-1)*(N-1)];
    for (int i = 0; i < N-1; i++) {
        for (int j = 0; j < N-1; j++) {
            int ind  = i*N+j;
            int ind_ = i*(N-1)+j;
            // Lower triangle
            elements[6*ind_  ] = ind;
            elements[6*ind_+1] = ind+N;
            elements[6*ind_+2] = ind+N+1;
            // Upper triangle
            elements[6*ind_+3] = ind;
            elements[6*ind_+4] = ind+1;
            elements[6*ind_+5] = ind+N+1;
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
    while(!glfwWindowShouldClose(window)) {
        glfwSwapBuffers(window);
        glfwPollEvents();

        // Clear the screen to black
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Timestepping
        for (int i = 0; i < skip; i++) {
            RungeKutta4(c, dt);
            t++;
        }

        // Update plot
        // begin = clock();
        for (int i = 0; i < N*N; i++) {
            colors[i] = (float) ((c[i] + 1.0)/2.0);
        }

        glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
        glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STREAM_DRAW);


        // Draw elements
        glDrawElements(GL_TRIANGLES, 6*(N-1)*(N-1), GL_UNSIGNED_INT, 0);

        // end = clock();
        // printf("Time = %f\n", (double)(end-begin)/CLOCKS_PER_SEC);
        // printf("Iter = %5d; Time = %.6f\n", t, t*dt);


        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GL_TRUE);
    }


    glDeleteProgram(shaderProgram);
    glDeleteShader(fragmentShader);
    glDeleteShader(vertexShader);

    glDeleteBuffers(1, &ebo);
    glDeleteBuffers(1, &vbo_pos);
    glDeleteBuffers(1, &vbo_colors);

    glDeleteVertexArrays(1, &vao);

    glfwDestroyWindow(window);
    glfwTerminate();
    free_functions();
    free(c);

    return EXIT_SUCCESS;
}
