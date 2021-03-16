extern "C" {
    #include "solver.h"
}

#define REAL 0
#define CPLX 1

size_t real_size = N_DISCR*N_DISCR*sizeof(double);
size_t cplx_size = N_DISCR*N_DISCR*sizeof(cufftDoubleComplex);

int Nblocks = (N_DISCR*N_DISCR)/256;
int Nthreads = 256;

double hh = 1.0 / (N_DISCR*N_DISCR);


/*
 *  Compute one iteration of Runge Kutta 4
 *  Return value is done in-place.
 */
void step(double* c, double dt){
    // K1
    f(c_gpu, k1);

    // K2
    k12_sum<<<Nblocks, Nthreads>>>(c_gpu, k1, tmp, dt);
    f(tmp, k2);

    // K3
    k12_sum<<<Nblocks, Nthreads>>>(c_gpu, k2, tmp, dt);
    f(tmp, k3);

    // K4
    k3_sum<<<Nblocks, Nthreads>>>(c_gpu, k3, tmp, dt);
    f(tmp, k4);

    // C_i+1
    k_sum_tot<<<Nblocks, Nthreads>>>(c_gpu, k1, k2, k3, k4, dt);
}

/*
 *  Compute the time derivative of c
 *  Return value is not in-place.
 */
void f(double* c, double* dc) {
    // Compute ĉ
    cufftExecD2Z(rfft, c, c_hat);

    // Compute ĉ³
    cube<<<Nblocks, Nthreads>>>(c, c_cube);
    cufftExecD2Z(rfft, c_cube, cval);

    // Compute F
    deriv<<<grid, threads>>>(c_hat, cval, hh);
    cufftExecZ2D(irfft, cval, dc);
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

    cudaMalloc((void **) &k1, real_size);
    cudaMalloc((void **) &k2, real_size);
    cudaMalloc((void **) &k3, real_size);
    cudaMalloc((void **) &k4, real_size);
    cudaMalloc((void **) &tmp, real_size);

    cudaMalloc((void **) &c_gpu, real_size);
    cudaMalloc((void **) &c_cube, real_size);

    cudaMalloc((void **) &c_hat, cplx_size);
    cudaMalloc((void **) &cval, cplx_size);

    cufftPlan2d(&rfft, N_DISCR, N_DISCR, CUFFT_D2Z);
    cufftPlan2d(&irfft, N_DISCR, N_DISCR, CUFFT_Z2D);

    // Initialise C
    cudaMemcpy(c_gpu, c, real_size, cudaMemcpyHostToDevice);
}

/*
 *  Free the various allocated arrays
 */
void free_solver() {
    cudaFree(k1);
    cudaFree(k2);
    cudaFree(k3);
    cudaFree(k4);
    cudaFree(tmp);

    cudaFree(c_gpu);
    cudaFree(c_cube);

    cudaFree(cval);
    cudaFree(c_hat);

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
__global__ void k12_sum(double* c, double* k, double* tmp, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    tmp[i] = c[i] + dt*k[i]/2.0;
}
__global__ void k3_sum(double* c, double* k, double* tmp, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    tmp[i] = c[i] + dt*k[i];
}
__global__ void k_sum_tot(double* c, double* k1, double* k2, double* k3, double* k4, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] += dt*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6.0;
}
__global__ void cube(double* c, double* cube) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    cube[i] = c[i]*c[i]*c[i];
}
__global__ void deriv(cufftDoubleComplex *c_hat, cufftDoubleComplex* cval, double hh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Wavenumber
    double l = (i < N_DISCR/2) ? i : i-N_DISCR;
    double k = - 4.0*M_PI*M_PI * (j*j + l*l);

    // Compute \hat{F}
    int ind = i*(N_DISCR/2+1)+j;
    cval[ind].x = hh*k * (cval[ind].x - c_hat[ind].x - 1e-4*k*c_hat[ind].x);
    cval[ind].y = hh*k * (cval[ind].y - c_hat[ind].y - 1e-4*k*c_hat[ind].y);
}
