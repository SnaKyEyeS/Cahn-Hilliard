#include "kernel.h"

#define REAL 0
#define CPLX 1


void cufft_laplacian(float* c, float h, float* delsq){
    size_t mem_size = N_DISCR*N_DISCR*sizeof(float);

    cudaMemcpy(rval_gpu, c, mem_size, cudaMemcpyHostToDevice);
    cufftExecR2C(plan, rval_gpu, cval_gpu, CUFFT_FORWARD);

    grid.x=8;
    grid.y=13;
    grid.z=1;
    threads.x=16;
    threads.y=5;
    threads.z=1;

    deriv<<<grid, threads>>>(h, cval_gpu);

    cufftExecC2R(plan, cval_gpu, rval_gpu, CUFFT_INVERSE);
    cudaMemcpy(delsq, rval_gpu, mem_size, cudaMemcpyDeviceToHost);
}

void init_cuda(){
    size_t mem_size = N_DISCR*N_DISCR*sizeof(float);
    size_t complex_size = N_DISCR*N_DISCR*sizeof(Complex);

    cudaMalloc((void **) &delsq_gpu, mem_size);
    cudaMalloc((void **) &rval_gpu, mem_size);
    cudaMalloc((void **) &cval_gpu, complex_size);

    cufftPlan2d(&rfft, N_DISCR, N_DISCR, CUFFT_R2C);
    cufftPlan2d(&irfft, N_DISCR, N_DISCR, CUFFT_C2R);
}

__global__ void deriv(float h, cufftComplex* cval){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int l, ind;
    float k;
    float factor = 4.0f*M_PI*M_PI*h*h;
    // Wavenumber
    l = (i < N_DISCR/2) ? i : i-N_DISCR;
    k = -factor * (j*j + l*l);

    // Multiply by (ik)²
    ind = i*(N_DISCR/2+1)+j;
    cval[ind][REAL] = k*cval[ind][REAL];
    cval[ind][CPLX] = k*cval[ind][CPLX];
}

void free_cuda(){
    cudaFree(delsq_gpu);
    cudaFree(rval_gpu);
    cudaFree(cval_gpu);
}
