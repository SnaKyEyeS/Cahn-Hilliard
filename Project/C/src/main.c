#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <stdio.h>
#include "functions.h"


// Shader sources
const GLchar* vertexSource = R"glsl(
    #version 150 core
    in vec2 position;
    in float color;
    out float Color;
    void main() {
        Color = color;
        gl_Position = vec4(position, 0.0, 1.0);
    }
)glsl";
const GLchar* fragmentSource = R"glsl(
    #version 150 core
    in float Color;
    out vec4 outColor;
    void main() {
        const vec4 kRedVec4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
        const vec4 kGreenVec4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
        const vec4 kBlueVec4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
        const vec2 kRedVec2 = vec2(-152.94239396, 59.28637943);
        const vec2 kGreenVec2 = vec2(4.27729857, 2.82956604);
        const vec2 kBlueVec2 = vec2(-89.90310912, 27.34824973);

        vec4 v4 = vec4( 1.0, Color, Color * Color, Color * Color * Color);
        vec2 v2 = v4.zw * v4.z;

        outColor = vec4(
            dot(v4, kRedVec4)   + dot(v2, kRedVec2),
            dot(v4, kGreenVec4) + dot(v2, kGreenVec2),
            dot(v4, kBlueVec4)  + dot(v2, kBlueVec2),
            1.0
        );
    }
)glsl";


int main() {

    // Init GLFW & window
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    GLFWwindow* window = glfwCreateWindow(800, 800, "OpenGL", NULL, NULL);
    glfwMakeContextCurrent(window);

    // Init GLEW
    glewExperimental = GL_TRUE;
    glewInit();

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

    // GLfloat positions[] = {
    //     -1.0f, -1.0f,
    //      1.0f, -1.0f,
    //      1.0f,  1.0f,
    //     -1.0f,  1.0f,
    // };

    GLfloat positions[2*N*N];
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int ind = i*N+j;
            positions[2*ind  ] = (float)(-1.0 + 2.0*i/N + 2.0*i/(N*N) + 2.0*i/(N*N*N) + 2.0*i/(N*N*N*N));
            positions[2*ind+1] = (float)(-1.0 + 2.0*j/N + 2.0*j/(N*N) + 2.0*j/(N*N*N) + 2.0*j/(N*N*N*N));
        }
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
    glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);

    // Specify vbo_pos' layout
    GLint posAttrib = glGetAttribLocation(shaderProgram, "position");
    glEnableVertexAttribArray(posAttrib);
    glVertexAttribPointer(posAttrib, 2, GL_FLOAT, GL_FALSE, 0, 0);

    // Create a Vertex Buffer Object for colors
    GLuint vbo_colors;
    glGenBuffers(2, &vbo_colors);

    GLfloat colors[] = {
        1.0f,
        0.5f,
        0.0f,
        0.0f,
    };

    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STREAM_DRAW);

    // Specify vbo_color's layout
    GLint colAttrib = glGetAttribLocation(shaderProgram, "color");
    glEnableVertexAttribArray(colAttrib);
    glVertexAttribPointer(colAttrib, 1, GL_FLOAT, GL_FALSE, 0, (void*)(0*sizeof(GLfloat)));

    // Create an Element Buffer Object and copy the element data to it
    GLuint ebo;
    glGenBuffers(1, &ebo);

    // GLuint elements[] = {
    //     0, N-1, 4*N+1,
    //     // 2, 3, 0,
    // };

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


    while(!glfwWindowShouldClose(window)) {
        glfwSwapBuffers(window);
        glfwPollEvents();

        // Clear the screen to black
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // colors[0] = (float)rand() / RAND_MAX;
        // colors[1] = (float)rand() / RAND_MAX;
        // colors[2] = (float)rand() / RAND_MAX;
        // colors[3] = (float)rand() / RAND_MAX;
        // glBindBuffer(GL_ARRAY_BUFFER, vbo_colors);
        // glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STREAM_DRAW);

        // Draw a triangle from the 3 vertices
        glDrawElements(GL_TRIANGLES, 6*(N-1)*(N-1)-3, GL_UNSIGNED_INT, 0);


        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, GL_TRUE);
    }


    glfwTerminate();
    return 0;
}
