#ifndef _WINDOW_H_
#define _WINDOW_H_

#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <stdbool.h>

bool drag;

GLFWwindow *init_window();

#endif
