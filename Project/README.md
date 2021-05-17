# Cahn-Hilliard solver

This project pertains to the numerical resolution of the Cahn-Hilliard equation in its dimensionless, constant mobility form:

<p align="center"><a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;c}{\partial&space;t}&space;=&space;\nabla^2\left(c^3&space;-&space;c&space;-&space;\kappa\nabla^2c\right)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;c}{\partial&space;t}&space;=&space;\nabla^2\left(c^3&space;-&space;c&space;-&space;\kappa\nabla^2c\right)" title="\frac{\partial c}{\partial t} = \nabla^2\left(c^3 - c - \kappa\nabla^2c\right)" /></a></p>

There are three different available implementations:
 * Python
 * C
 * CUDA

## Python implementation

The files can be found in the [python](python/) subfolder. The installation requirements are fairly simple and we recommend the following procedure:
``` bash
# Create venv & install requirements
cd /path/to/python
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt

# Run one of the scripts
source venv/bin/activate
python spectral.py
```

There are three python implementations:
 * [naive.py](python/naive.py) using finite differences & an explicit euler time stepping scheme
 * [spectral.py](python/spectral.py) using Fourier spectral differentiation methods and various more advanced solvers (such as IMEX and ETDRK4).
 * [vmobility.py](python/spectral.py) implementing the Cahn-Hilliard equation with variable mobility M(c) = (1-ac²).

Should you wish to save a movie of the animation, you can easily do so by changing the `if False:` into `if True:` in the main subroutine of the scripts.

## C/CUDA implementations

The files can be found in the [C](C/) subfolder. The installation requirements will be detailled later, as it is a bit more complex. First, we will detail the compilation & running procedure:
```bash
cd /path/to/C
mkdir build
cd build
cmake ..
make
./project
```
When successful, a window should appear with the animation of the Cahn-Hilliard solver. It is possible to choose different compilation parameters to change the simulation parameters or choose different solvers. Those parameters are flags you may add to the `cmake` command:
 * `-DN_DISCR=128|256|512|...` (default is 128): sets the discretisation level of the N-by-N grid. To avoid any problem (and for the sake of efficiency), this number should always be a power of 2 and greater or equal to 128, though it should work for any integer as long as CUDA is turned off.
  * `-DSOLVER=ETDRK4|IMEX` (default is ETDRK4): changes the time stepping scheme. ETDRK4 is a bit slower per iteration than the IMEX schemes, but otherwise way more precise and much faster to reach a given precision and has a bigger stability region. Therefore, it should always be preferred.
  * `-DUSE_CUDA=off|on` (default is OFF): whether to use CUDA to accelerate the simulation or not. Only available if you have a Nvidia GPU and CUDA installed. As it is much faster, it should be used whenever possible.
  * `-DMOBILITY=CONSTANT|VARIABLE` (default is CONSTANT): switch to variable mobility simulation. The simulation parameters then change accordingly.

### Installation instructions

The following installation instructions are specifically made for the `apt` package manager, but should be equally easy to follow with other package managers with a bit of Googling skills. However, we do not provide any help for Windows (the best help  you may find is to install a Linux-based OS...).

##### Visualisation (OpenGL & GLFW3)
The visualisation is made using OpenGL & GLFW3. You will need to make sure you have OpenGL, GLFW3 and GLEW installed on your system:
```bash
sudo apt-get install freeglut3-dev libglfw3-dev  libglew-dev
```

##### Pure-C code: FFT libraries
You will need to install one of the two following libraries to perform the FFT's necessary to run the code. You can choose either Intel's MKL, which may be installed using the following:
```bash
sudo apt-get install intel-mkl-full
```
or install [FFTW3](http://www.fftw.org/download.html):
```bash
sudo apt-get install libfftw3-dev
```
By default, the compiler will choose Intel's MKL if both libraries are available as it proved a bit more efficient (at least on our computers, which are both Intel-based processors).

##### CUDA
The CUDA version of the solver uses cuFFT to perform the FFT's and uses CUDA kernels to accelerate various computations. To run this version, you will need a Nvidia GPU as well as to install CUDA. As the installation procedure is a bit more tedious, we recommend you to follow this [official installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
