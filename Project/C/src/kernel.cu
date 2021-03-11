extern "C" {
    #include "kernel.h"
}

#define REAL 0
#define CPLX 1


void cufft_laplacian(double* c, double h, double* delsq) {
    size_t mem_size = N_DISCR*N_DISCR*sizeof(double);

    cudaMemcpy(rval_gpu, c, mem_size, cudaMemcpyHostToDevice);
    cufftExecD2Z(rfft, rval_gpu, cval_gpu);

    grid.x=8;
    grid.y=13;
    grid.z=1;
    threads.x=16;
    threads.y=5;
    threads.z=1;

    deriv<<<grid, threads>>>(h, cval_gpu);

    cufftExecZ2D(irfft, cval_gpu, rval_gpu);
    cudaMemcpy(delsq, rval_gpu, mem_size, cudaMemcpyDeviceToHost);
}

void init_cuda() {
    size_t mem_size = N_DISCR*N_DISCR*sizeof(double);
    size_t complex_size = N_DISCR*N_DISCR*sizeof(cufftDoubleComplex);

    cudaMalloc((void **) &delsq_gpu, mem_size);
    cudaMalloc((void **) &rval_gpu, mem_size);
    cudaMalloc((void **) &cval_gpu, complex_size);

    cufftPlan2d(&rfft, N_DISCR, N_DISCR, CUFFT_D2Z);
    cufftPlan2d(&irfft, N_DISCR, N_DISCR, CUFFT_Z2D);
}

__global__ void deriv(double h, cufftDoubleComplex* cval) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int l, ind;
    double k;
    double factor = 4.0*M_PI*M_PI*h*h;
    // Wavenumber
    l = (i < N_DISCR/2) ? i : i-N_DISCR;
    k = -factor * (j*j + l*l);

    // Multiply by (ik)²
    ind = i*(N_DISCR/2+1)+j;
    cval[ind].x = k*cval[ind].x;
    cval[ind].y = k*cval[ind].y;
}

void free_cuda() {
    cudaFree(delsq_gpu);
    cudaFree(rval_gpu);
    cudaFree(cval_gpu);
}
