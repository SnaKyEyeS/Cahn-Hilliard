#include "functions.h"


/*
 *  Compute one iteration of Runge Kutta 4
 *  Return value is done in-place.
 */
void RungeKutta4(double* c, double dt){
    // K1
    f(c, k1);

    // K2
    for(int i = 0; i < N*N; i++) {
        tmp[i] = c[i] + dt*k1[i]/2.0;
    }
    f(tmp, k2);

    // K3
    for(int i = 0; i < N*N; i++) {
        tmp[i] = c[i] + dt*k2[i]/2.0;
    }
    f(tmp, k3);

    // K4
    for(int i = 0; i < N*N; i++) {
        tmp[i] = c[i] + dt*k3[i];
    }
    f(tmp, k4);

    // C_i+1
    for(int i = 0; i < N*N; i++) {
        c[i] += dt*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6.0;
    }
}

/*
 *  Compute the time derivative of c
 *  Return value is not in-place.
 */
void f(double* c, double* dc) {

    laplacian(c, 1.0/N, delsq);
    for(int i = 0; i < N*N; i++) {
        delsq[i] = c[i]*c[i]*c[i] - c[i] - A*A*delsq[i];
    }
    laplacian(delsq, 1.0/N, dc);
}

/*
 *  Compute the 2-D, cartesian Laplacian of c
 *  Return value is not in-place.
 */
void laplacian(double* c, double h, double* delsq){

    // Forward 2D real-valued FFT
    memcpy(rval, c, N*N*sizeof(double));
    fftw_execute(rfft2);

    // Take the derivative
    int l, ind;
    double k;
    double factor = 4*M_PI*M_PI*h*h;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N/2+1; j++) {

            // Wavenumber
            l = (i < N/2) ? i : i-N;
            k = -factor * (j*j + l*l);

            // Multiply by (ik)²
            ind = i*(N/2+1)+j;
            cval[ind][REAL] = k*cval[ind][REAL];
            cval[ind][CPLX] = k*cval[ind][CPLX];
        }
    }

    // Backward 2D real-valued FFT
    fftw_execute(irfft2);
    memcpy(delsq, rval, N*N*sizeof(double));
}

/*
 *  Initialise the various stuff
 */
void init_functions() {
    k1    = (double*) malloc(N*N*sizeof(double));
    k2    = (double*) malloc(N*N*sizeof(double));
    k3    = (double*) malloc(N*N*sizeof(double));
    k4    = (double*) malloc(N*N*sizeof(double));
    tmp   = (double*) malloc(N*N*sizeof(double));
    delsq = (double*) malloc(N*N*sizeof(double));

    cval = fftw_alloc_complex(N*(N/2+1));
    rval = fftw_alloc_real(N*N);

    rfft2  = fftw_plan_dft_r2c_2d(N, N, rval, cval, FFTW_PATIENT);
    irfft2 = fftw_plan_dft_c2r_2d(N, N, cval, rval, FFTW_PATIENT);
}

/*
 *  Free the various allocated arrays
 */
void free_functions() {
    // RungeKutta4
    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(tmp);

    // f
    free(delsq);

    // laplacian
    free(cval);
    free(rval);
    fftw_destroy_plan(rfft2);
    fftw_destroy_plan(irfft2);
    fftw_cleanup();
}
