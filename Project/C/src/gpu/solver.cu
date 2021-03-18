extern "C" {
    #include "solver.h"
    #include <stdio.h>
}

#define REAL 0
#define CPLX 1

size_t real_size = N_DISCR*N_DISCR*sizeof(double);
size_t cplx_size = N_DISCR*(1+N_DISCR/2)*sizeof(cufftDoubleComplex);

int Nblocks = (N_DISCR*N_DISCR)/256;
int Nthreads = 256;

double hh = 1.0 / (N_DISCR*N_DISCR);


/*
 *  Compute one iteration of Runge Kutta 4
 *  Return value is done in-place.
 */
int iter = 1;
double *c_gpu;
double *c_cube;
cufftDoubleComplex *tmp;
cufftDoubleComplex *c_hat;
cufftDoubleComplex *c_hat_prev;
cufftDoubleComplex *f_hat;
cufftDoubleComplex *f_hat_prev;

void step(double* c, double dt) {
    // Initialise solver; perform first iteration
    if (iter == 1) {
        // Compute ĉ
        cufftExecD2Z(rfft, c_gpu, c_hat_prev);

        // Compute ĉ³ - ĉ
        cube<<<Nblocks, Nthreads>>>(c_gpu, c_cube);
        cufftExecD2Z(rfft, c_cube, f_hat_prev);

        // Compute c_1
        first_order<<<grid, threads>>>(c_hat_prev, f_hat_prev, dt, hh, tmp);
        cufftExecZ2D(irfft, tmp, c_gpu);

        iter++;
    }

    // Compute ĉ
    cufftExecD2Z(rfft, c_gpu, c_hat);

    // Compute ĉ³ - ĉ
    cube<<<Nblocks, Nthreads>>>(c_gpu, c_cube);
    cufftExecD2Z(rfft, c_cube, f_hat);

    // Compute c_{i+1}
    second_order<<<grid, threads>>>(c_hat, c_hat_prev, f_hat, f_hat_prev, dt, hh, tmp);
    cufftExecZ2D(irfft, tmp, c_gpu);

    // Save variables for next iteration
    cudaMemcpy(c_hat_prev, c_hat, cplx_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(f_hat_prev, f_hat, cplx_size, cudaMemcpyDeviceToDevice);
}

/*
 *  Initialise the various stuff
 */
void init_solver(double *c) {
    grid.x = N_DISCR/128;
    grid.y = 1 + N_DISCR/2;
    grid.z = 1;
    threads.x = 128;
    threads.y = 1;
    threads.z = 1;

    // Semi-implicit scheme
    cudaMalloc((void **) &c_gpu,      real_size);
    cudaMalloc((void **) &c_cube,     real_size);
    cudaMalloc((void **) &tmp,        cplx_size);
    cudaMalloc((void **) &c_hat,      cplx_size);
    cudaMalloc((void **) &c_hat_prev, cplx_size);
    cudaMalloc((void **) &f_hat,      cplx_size);
    cudaMalloc((void **) &f_hat_prev, cplx_size);

    // cuFFT
    cufftPlan2d(&rfft,  N_DISCR, N_DISCR, CUFFT_D2Z);
    cufftPlan2d(&irfft, N_DISCR, N_DISCR, CUFFT_Z2D);

    // Initialise C
    cudaMemcpy(c_gpu, c, real_size, cudaMemcpyHostToDevice);
}

/*
 *  Free the various allocated arrays
 */
void free_solver() {

    cudaFree(c_gpu);
    cudaFree(c_cube);
    cudaFree(c_hat);
    cudaFree(c_hat_prev);
    cudaFree(f_hat);
    cudaFree(f_hat_prev);

    cufftDestroy(rfft);
    cufftDestroy(irfft);
}

/*
 *  Copy solution from Device to Host
 */
void cudaGetSolution(double *c) {
    cudaMemcpy(c, c_gpu, real_size, cudaMemcpyDeviceToHost);
}

/*
 *  Kernel stuff
 */
__global__ void cube(double* c, double* cube) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    cube[i] = c[i]*c[i]*c[i] - c[i];
}
__global__ void first_order(cufftDoubleComplex *c_hat, cufftDoubleComplex* f_hat, double dt, double hh, cufftDoubleComplex *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Wavenumber
    double l = (i < N_DISCR/2) ? i : i-N_DISCR;
    double k = 4.0*M_PI*M_PI * (j*j + l*l);

    // Compute \hat{F}
    int ind = i*(N_DISCR/2+1)+j;
    out[ind].x = hh * (c_hat[ind].x - dt*k*f_hat[ind].x) / (1.0 + dt*1e-4*k*k);
    out[ind].y = hh * (c_hat[ind].y - dt*k*f_hat[ind].y) / (1.0 + dt*1e-4*k*k);
}
__global__ void second_order(cufftDoubleComplex *c_hat, cufftDoubleComplex* c_hat_prev, cufftDoubleComplex* f_hat, cufftDoubleComplex* f_hat_prev, double dt, double hh, cufftDoubleComplex *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Wavenumber
    double l = (i < N_DISCR/2) ? i : i-N_DISCR;
    double k = 4.0*M_PI*M_PI * (j*j + l*l);

    // Compute \hat{F}
    int ind = i*(N_DISCR/2+1)+j;
    out[ind].x = hh*(4.0*c_hat[ind].x - c_hat_prev[ind].x - 2.0*dt*k*(2.0*f_hat[ind].x - f_hat_prev[ind].x)) / (3.0 + 2e-4*dt*k*k);
    out[ind].y = hh*(4.0*c_hat[ind].y - c_hat_prev[ind].y - 2.0*dt*k*(2.0*f_hat[ind].y - f_hat_prev[ind].y)) / (3.0 + 2e-4*dt*k*k);
}
