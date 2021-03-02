#version 150 core
in vec2 pos;
in float value;
out float vertexvalue;
void main() { 
  gl_Position = vec4(pos, 0.0, 1.0);
  vertexvalue = value;
}
