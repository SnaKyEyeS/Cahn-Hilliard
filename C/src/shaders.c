#include "shaders.h"


void init_shaders() {
    // Vertex shader
    vertexSource = R"glsl(
        #version 450 core

        in vec2 position;
        in float color;
        out VS_OUT {
            float color;
        } vs_out;

        void main() {
            vs_out.color = color;
            gl_Position = vec4(position, 0.0, 1.0);
        }
    )glsl";

    // Geometry shader
    geometrySource = R"glsl(
        #version 450 core

        layout (lines_adjacency) in;
        layout (triangle_strip, max_vertices = 4) out;

        in VS_OUT {
            float color;
        } gs_in[];
        out GS_OUT {
            float color;
        } gs_out;

        void main() {
            for (int i = 0; i < 4; i++) {
                gl_Position = gl_in[i].gl_Position;
                gs_out.color = gs_in[i].color;
                EmitVertex();
            }

            EndPrimitive();
        }
    )glsl";

    // Fragment shader
    fragmentSource = R"glsl(
        #version 450 core

        in GS_OUT {
            float color;
        } fs_in;
        out vec4 color;

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
            color = turbo(fs_in.color);
        }
    )glsl";

    // Create and compile the vertex shader
    vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexSource, NULL);
    glCompileShader(vertexShader);

    // Create and compile the geometry shader
    geometryShader = glCreateShader(GL_GEOMETRY_SHADER);
    glShaderSource(geometryShader, 1, &geometrySource, NULL);
    glCompileShader(geometryShader);

    // Create and compile the fragment shader
    fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
    glCompileShader(fragmentShader);

    // Link the vertex and fragment shader into a shader program
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, geometryShader);
    glAttachShader(shaderProgram, fragmentShader);
    glBindFragDataLocation(shaderProgram, 0, "color");
    glLinkProgram(shaderProgram);
    glUseProgram(shaderProgram);
}

void free_shaders() {
    glDeleteProgram(shaderProgram);
    glDeleteShader(fragmentShader);
    glDeleteShader(geometryShader);
    glDeleteShader(vertexShader);
}
