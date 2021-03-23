extern "C" {
    #include "solver.h"
    #include <stdio.h>
}

#define REAL 0
#define CPLX 1
#define FOUR_PI_SQUARED 39.478417604357432


size_t real_size = N_DISCR*N_DISCR*sizeof(double);
size_t cplx_size = N_DISCR*(1+N_DISCR/2)*sizeof(complex);

int Nblocks = (N_DISCR*N_DISCR)/256;
int Nthreads = 256;

double hh = 1.0 / (N_DISCR*N_DISCR);


/*
 *  Compute one iteration of Runge Kutta 4
 *  Return value is done in-place.
 */
int iter = 1;
double *c_gpu, *c_cube;
complex *tmp, *out;
complex *c_hat, *c_hat_1;
complex *f_hat, *f_hat_1;

void step(double dt) {
    // Compute ĉ
    cufftExecD2Z(rfft, c_gpu, c_hat);

    // Compute ĉ³ - ĉ
    cube<<<Nblocks, Nthreads>>>(c_gpu, c_cube);
    cufftExecD2Z(rfft, c_cube, f_hat);

    // Compute ĉ_i+1
    if (iter == 1) {            // IMEX-BDF1
        imex_bdf1<<<grid, threads>>>(c_hat, f_hat, dt, hh, out);

    } else {                    // IMEX-BDF2
        imex_bdf2<<<grid, threads>>>(c_hat, c_hat_1, f_hat, f_hat_1, dt, hh, out);
    }

    // Back to physical domain
    cufftExecZ2D(irfft, out, c_gpu);

    // Save variables for next iteration
    tmp = c_hat_1;
    c_hat_1 = c_hat;
    c_hat = tmp;

    tmp = f_hat_1;
    f_hat_1 = f_hat;
    f_hat = tmp;

    iter++;
}

/*
 *  Initialise the various stuff
 */
void init_solver(double *c, double dt) {
    switch(SOLVER) {
        case RK4:
            printf("Selected solver is RK4\n");
            break;

        case IMEX:
            printf("Selected solver is IMEX\n");
            break;

        case ETDRK4:
            printf("Selected solver is ETDRK4\n");
            break;
    }


    grid.x = N_DISCR/128;
    grid.y = 1 + N_DISCR/2;
    grid.z = 1;
    threads.x = 128;
    threads.y = 1;
    threads.z = 1;

    // Semi-implicit scheme
    cudaMalloc((void **) &c_gpu,   real_size);
    cudaMalloc((void **) &c_cube,  real_size);
    cudaMalloc((void **) &out,     cplx_size);
    cudaMalloc((void **) &c_hat,   cplx_size);
    cudaMalloc((void **) &c_hat_1, cplx_size);
    cudaMalloc((void **) &f_hat,   cplx_size);
    cudaMalloc((void **) &f_hat_1, cplx_size);

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
    cudaFree(out);
    cudaFree(c_hat);
    cudaFree(c_hat_1);
    cudaFree(f_hat);
    cudaFree(f_hat_1);

    cufftDestroy(rfft);
    cufftDestroy(irfft);
}

/*
 *  Copy solution from Device to Host
 */
void getSolution(double *c) {
    cudaMemcpy(c, c_gpu, real_size, cudaMemcpyDeviceToHost);
}

/*
 *  Kernel stuff
 */
__global__ void cube(double* c, double* cube) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    cube[i] = c[i]*c[i]*c[i] - c[i];
}
__global__ void imex_bdf1(complex *c_hat, complex* f_hat, double dt, double hh, complex *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Wavenumber
    double l = (i < N_DISCR/2) ? i : i-N_DISCR;
    double k = FOUR_PI_SQUARED * (j*j + l*l);

    // Compute \hat{F}
    int ind = i*(N_DISCR/2+1)+j;
    out[ind].x = hh * (c_hat[ind].x - dt*k*f_hat[ind].x) / (1.0 + dt*1e-4*k*k);
    out[ind].y = hh * (c_hat[ind].y - dt*k*f_hat[ind].y) / (1.0 + dt*1e-4*k*k);
}
__global__ void imex_bdf2(complex *c_hat, complex* c_hat_1, complex* f_hat, complex* f_hat_1, double dt, double hh, complex *out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Wavenumber
    double l = (i < N_DISCR/2) ? i : i-N_DISCR;
    double k = FOUR_PI_SQUARED * (j*j + l*l);

    // Compute \hat{F}
    int ind = i*(N_DISCR/2+1)+j;
    out[ind].x = hh*(4.0*c_hat[ind].x - c_hat_1[ind].x - 2.0*dt*k*(2.0*f_hat[ind].x - f_hat_1[ind].x)) / (3.0 + 2e-4*dt*k*k);
    out[ind].y = hh*(4.0*c_hat[ind].y - c_hat_1[ind].y - 2.0*dt*k*(2.0*f_hat[ind].y - f_hat_1[ind].y)) / (3.0 + 2e-4*dt*k*k);
}
