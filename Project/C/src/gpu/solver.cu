extern "C" {
    #include "solver.h"
    #include "helper.h"
    #include <stdio.h>
}

#define REAL 0
#define CPLX 1
#define FOUR_PI_SQUARED 39.478417604357432


size_t real_size = N_DISCR*N_DISCR*sizeof(double);
size_t cplx_size = N_DISCR*(1+N_DISCR/2)*sizeof(complex);

dim3 grid, threads;
int NblocksReal  = N_DISCR*N_DISCR/256;
int NthreadsReal = 256;
int NblocksCplx  = N_DISCR*(1+N_DISCR/2)/128;
int NthreadsCplx = 128;

double hh = 1.0 / (N_DISCR*N_DISCR);


/*
 *  Compute one iteration of Runge Kutta 4
 *  Return value is done in-place.
 */
double *c_gpu;
complex *c_hat, *out;

void step(double dt) {
    switch (SOLVER) {
        case IMEX:
            imex(dt);
            break;

        case ETDRK4:
            etdrk4(dt);
            break;
    }
}

/*
 *  IMEX solver.
 */
int iter = 1;
complex *tmp;
complex *c_hat_0, *c_hat_1;
complex *f_hat_0, *f_hat_1;

void imex(double dt) {
    // Save current iteration
    tmp = c_hat_0;
    c_hat_0 = c_hat;
    c_hat = tmp;

    // Compute ĉ³ - ĉ
    non_linear_term(c_hat_0, f_hat_0);

    // Apply IMEX scheme
    if (iter == 1) {            // IMEX-BDF1
        imex_bdf1<<<grid, threads>>>(c_hat_0, f_hat_0, dt, hh, c_hat);

    } else {                    // IMEX-BDF2
        imex_bdf2<<<grid, threads>>>(c_hat_0, c_hat_1, f_hat_0, f_hat_1, dt, hh, c_hat);
    }

    // Save variables for next iteration
    tmp = c_hat_1;
    c_hat_1 = c_hat_0;
    c_hat_0 = tmp;

    tmp = f_hat_1;
    f_hat_1 = f_hat_0;
    f_hat_0 = tmp;

    iter++;
}

__global__ void imex_bdf1(complex *c_hat_0, complex* f_hat_0, double dt, double hh, complex *c_hat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Wavenumber
    double l = (i < N_DISCR/2) ? i : i-N_DISCR;
    double k = FOUR_PI_SQUARED * (j*j + l*l);

    // Compute next ĉ_{i+1}
    int ind = i*(N_DISCR/2+1)+j;
    c_hat[ind].x = (c_hat_0[ind].x + dt*f_hat_0[ind].x) / (1.0 + dt*1e-4*k*k);
    c_hat[ind].y = (c_hat_0[ind].y + dt*f_hat_0[ind].y) / (1.0 + dt*1e-4*k*k);
}
__global__ void imex_bdf2(complex *c_hat_0, complex* c_hat_1, complex* f_hat_0, complex* f_hat_1, double dt, double hh, complex *c_hat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Wavenumber
    double l = (i < N_DISCR/2) ? i : i-N_DISCR;
    double k = FOUR_PI_SQUARED * (j*j + l*l);

    // Compute \hat{F}
    int ind = i*(N_DISCR/2+1)+j;
    c_hat[ind].x = (4.0*c_hat_0[ind].x - c_hat_1[ind].x + 2.0*dt*(2.0*f_hat_0[ind].x - f_hat_1[ind].x)) / (3.0 + 2e-4*dt*k*k);
    c_hat[ind].y = (4.0*c_hat_0[ind].y - c_hat_1[ind].y + 2.0*dt*(2.0*f_hat_0[ind].y - f_hat_1[ind].y)) / (3.0 + 2e-4*dt*k*k);
}

/*
 *  ETDRK solver.
 */
double *e1, *e2, *f1, *f2, *f3, *q;
complex *fa, *fb, *fc, *Nu, *Na, *Nb, *Nc;

void etdrk4(double dt) {
    // Compute N(u)
    non_linear_term(c_hat, Nu);

    // Compute fa & N(a)
    compute_fa<<<NblocksCplx, NthreadsCplx>>>(c_hat, Nu, e2, q, fa);
    non_linear_term(fa, Na);

    // Compute fb & N(a)
    compute_fb<<<NblocksCplx, NthreadsCplx>>>(c_hat, Na, e2, q, fb);
    non_linear_term(fb, Nb);

    // Compute a & N(a)
    compute_fc<<<NblocksCplx, NthreadsCplx>>>(fa, Nu, Nb, e2, q, fc);
    non_linear_term(fc, Nc);

    // Compute ĉ_{i+1}
    etdrk4_next<<<NblocksCplx, NthreadsCplx>>>(c_hat, Nu, Na, Nb, Nc, e1, f1, f2, f3);
}

__global__ void compute_fa(complex *c_hat, complex *Nu, double *e2, double *q, complex *fa) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    fa[i].x = e2[i]*c_hat[i].x + q[i]*Nu[i].x;
    fa[i].y = e2[i]*c_hat[i].y + q[i]*Nu[i].y;
}
__global__ void compute_fb(complex *c_hat, complex *Na, double *e2, double *q, complex *fb) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    fb[i].x = e2[i]*c_hat[i].x + q[i]*Na[i].x;
    fb[i].y = e2[i]*c_hat[i].y + q[i]*Na[i].y;
}
__global__ void compute_fc(complex *fa, complex *Nu, complex *Nb, double *e2, double *q, complex *fc) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    fc[i].x = e2[i]*fa[i].x + q[i]*(2.0*Nb[i].x - Nu[i].x);
    fc[i].y = e2[i]*fa[i].y + q[i]*(2.0*Nb[i].y - Nu[i].y);
}
__global__ void etdrk4_next(complex* c_hat, complex *Nu, complex *Na, complex *Nb, complex *Nc, double *e1, double *f1, double *f2, double* f3) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    c_hat[i].x = e1[i]*c_hat[i].x + f1[i]*Nu[i].x + 2.0*f2[i]*(Na[i].x + Nb[i].x) + f3[i]*Nc[i].x;
    c_hat[i].y = e1[i]*c_hat[i].y + f1[i]*Nu[i].y + 2.0*f2[i]*(Na[i].y + Nb[i].y) + f3[i]*Nc[i].y;
}


/*
 *  Compute -k*F(c³ -c) where F is the Fourier transform.
 */
void non_linear_term(complex *c_hat, complex *f_hat) {
    scale<<<NblocksCplx, NthreadsCplx>>>(c_hat, f_hat, hh);
    cufftExecZ2D(irfft, f_hat, c_gpu);
    f<<<NblocksReal, NthreadsReal>>>(c_gpu, c_gpu);
    cufftExecD2Z(rfft, c_gpu, f_hat);
    deriv<<<grid, threads>>>(f_hat);
}

__global__ void scale(complex *c_hat, complex *out, double hh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i].x = c_hat[i].x * hh;
    out[i].y = c_hat[i].y * hh;
}
__global__ void f(double *c, double *f) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    f[i] = c[i] - c[i]*c[i]*c[i];
}
__global__ void deriv(complex *c_hat) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Wavenumber
    double l = (i < N_DISCR/2) ? i : i-N_DISCR;
    double k = FOUR_PI_SQUARED * (j*j + l*l);

    // Compute the derivative
    int ind = i*(N_DISCR/2+1)+j;
    c_hat[ind].x *= k;
    c_hat[ind].y *= k;
}



/*
 *  Initialise the various stuff
 */
void init_solver(double *c, double dt) {
    switch(SOLVER) {
        case IMEX:
            cudaMalloc((void **) &c_hat_0, cplx_size);
            cudaMalloc((void **) &c_hat_1, cplx_size);
            cudaMalloc((void **) &f_hat_0, cplx_size);
            cudaMalloc((void **) &f_hat_1, cplx_size);
            break;

        case ETDRK4:
            int nCplxElem = N_DISCR*(N_DISCR/2+1);
            double *e1_cpu = (double*) malloc(6*nCplxElem*sizeof(double));
            double *e2_cpu = &e1_cpu[  nCplxElem];
            double *f1_cpu = &e1_cpu[2*nCplxElem];
            double *f2_cpu = &e1_cpu[3*nCplxElem];
            double *f3_cpu = &e1_cpu[4*nCplxElem];
            double *q_cpu  = &e1_cpu[5*nCplxElem];

            init_etdrk4(e1_cpu, e2_cpu, f1_cpu, f2_cpu, f3_cpu, q_cpu, dt);

            cudaMalloc((void **) &e1, nCplxElem*sizeof(double));
            cudaMalloc((void **) &e2, nCplxElem*sizeof(double));
            cudaMalloc((void **) &f1, nCplxElem*sizeof(double));
            cudaMalloc((void **) &f2, nCplxElem*sizeof(double));
            cudaMalloc((void **) &f3, nCplxElem*sizeof(double));
            cudaMalloc((void **) &q , nCplxElem*sizeof(double));

            cudaMemcpy(e1, e1_cpu, nCplxElem*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(e2, e2_cpu, nCplxElem*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(f1, f1_cpu, nCplxElem*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(f2, f2_cpu, nCplxElem*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(f3, f3_cpu, nCplxElem*sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(q , q_cpu , nCplxElem*sizeof(double), cudaMemcpyHostToDevice);

            free(e1_cpu);

            cudaMalloc((void **) &fa, cplx_size);
            cudaMalloc((void **) &fb, cplx_size);
            cudaMalloc((void **) &fc, cplx_size);
            cudaMalloc((void **) &Nu, cplx_size);
            cudaMalloc((void **) &Na, cplx_size);
            cudaMalloc((void **) &Nb, cplx_size);
            cudaMalloc((void **) &Nc, cplx_size);
            break;
    }

    // Complex grid
    grid.x = N_DISCR/128;
    grid.y = 1 + N_DISCR/2;
    grid.z = 1;
    threads.x = 128;
    threads.y = 1;
    threads.z = 1;

    // Input & output
    cudaMalloc((void **) &c_gpu,   real_size);
    cudaMalloc((void **) &out,     cplx_size);
    cudaMalloc((void **) &c_hat,   cplx_size);

    // cuFFT
    cufftPlan2d(&rfft,  N_DISCR, N_DISCR, CUFFT_D2Z);
    cufftPlan2d(&irfft, N_DISCR, N_DISCR, CUFFT_Z2D);

    // Initialise C
    cudaMemcpy(c_gpu, c, real_size, cudaMemcpyHostToDevice);
    cufftExecD2Z(rfft, c_gpu, c_hat);
}

/*
 *  Free the various allocated arrays
 */
void free_solver() {
    switch (SOLVER) {
        case IMEX:
            cudaFree(c_hat_0);
            cudaFree(c_hat_1);
            cudaFree(f_hat_0);
            cudaFree(f_hat_1);
            break;

        case ETDRK4:
            cudaFree(e1);
            cudaFree(e2);
            cudaFree(f1);
            cudaFree(f2);
            cudaFree(f3);
            cudaFree(q );

            cudaFree(Nu);
            cudaFree(Na);
            cudaFree(Nb);
            cudaFree(Nc);
            cudaFree(fa);
            cudaFree(fb);
            cudaFree(fc);
            break;
    }

    // Input & output
    cudaFree(c_gpu);
    cudaFree(c_hat);
    cudaFree(out);

    // cuFFT
    cufftDestroy(rfft);
    cufftDestroy(irfft);
}

/*
 *  Copy solution from Device to Host
 */
void getSolution(double *c) {
    scale<<<NblocksCplx, NthreadsCplx>>>(c_hat, out, hh);
    cufftExecZ2D(irfft, out, c_gpu);
    cudaMemcpy(c, c_gpu, real_size, cudaMemcpyDeviceToHost);
}
