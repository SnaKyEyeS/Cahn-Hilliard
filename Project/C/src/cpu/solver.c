#include "solver.h"

#define REAL 0
#define CPLX 1

double hh = 1.0 / (N_DISCR*N_DISCR);
int nRealElem = N_DISCR*N_DISCR;
int nCplxElem = N_DISCR*(N_DISCR/2+1);


/*
 *  Compute one iteration of Runge Kutta 4
 *  Return value is done in-place.
 */
void RungeKutta4(double* c, double dt){
    // K1
    f(c, k1);

    // K2
    for(int i = 0; i < nRealElem; i++) {
        tmp[i] = c[i] + dt*k1[i]/2.0;
    }
    f(tmp, k2);

    // K3
    for(int i = 0; i < nRealElem; i++) {
        tmp[i] = c[i] + dt*k2[i]/2.0;
    }
    f(tmp, k3);

    // K4
    for(int i = 0; i < nRealElem; i++) {
        tmp[i] = c[i] + dt*k3[i];
    }
    f(tmp, k4);

    // C_i+1
    for(int i = 0; i < nRealElem; i++) {
        c[i] += dt*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6.0;
    }
}

/*
 *  Compute the time derivative of c
 *  Return value is not in-place.
 */
void f(double* c, double* dc) {
    // Compute ĉ
    memcpy(rval, c, nRealElem*sizeof(double));
    fftw_execute(rfft2);
    memcpy(c_hat, cval, nCplxElem*sizeof(fftw_complex));

    // Compute ĉ³
    for(int i = 0; i < nRealElem; i++) {
        rval[i] = c[i]*c[i]*c[i];
    }
    fftw_execute(rfft2);

    // Compute F
    for(int i = 0; i < nCplxElem; i++) {
        cval[i][REAL] = k[i] * (cval[i][REAL] - c_hat[i][REAL] - 1e-4*k[i]*c_hat[i][REAL]);
        cval[i][CPLX] = k[i] * (cval[i][CPLX] - c_hat[i][CPLX] - 1e-4*k[i]*c_hat[i][CPLX]);
    }
    fftw_execute(irfft2);
    for (int i = 0; i < nRealElem; i++) {
        dc[i] = hh*rval[i];
    }
}

/*
 *  Initialise the various stuff
 */
void init_solver(double *c) {
    // RK4
    k1    = (double*) malloc(nRealElem*sizeof(double));
    k2    = (double*) malloc(nRealElem*sizeof(double));
    k3    = (double*) malloc(nRealElem*sizeof(double));
    k4    = (double*) malloc(nRealElem*sizeof(double));
    tmp   = (double*) malloc(nRealElem*sizeof(double));

    // F
    c_hat = fftw_alloc_complex(nCplxElem);
    k = (double*) malloc(nCplxElem*sizeof(double));
    double factor = -4*M_PI*M_PI;
    for (int i = 0; i < N_DISCR; i++) {
        for (int j = 0; j < N_DISCR/2+1; j++) {
            int l = (i < N_DISCR/2) ? i : i-N_DISCR;
            k[i*(N_DISCR/2+1)+j] = factor*(j*j + l*l);
        }
    }

    // FFTW
    cval = fftw_alloc_complex(nCplxElem);
    rval = fftw_alloc_real(nRealElem);
    rfft2  = fftw_plan_dft_r2c_2d(N_DISCR, N_DISCR, rval, cval, FFTW_PATIENT);
    irfft2 = fftw_plan_dft_c2r_2d(N_DISCR, N_DISCR, cval, rval, FFTW_PATIENT);
}

/*
 *  Free the various allocated arrays
 */
void free_solver() {
    // RK4
    free(k1);
    free(k2);
    free(k3);
    free(k4);
    free(tmp);

    // F
    free(k);
    fftw_free(c_hat);

    // FFTW
    fftw_free(cval);
    fftw_free(rval);
    fftw_destroy_plan(rfft2);
    fftw_destroy_plan(irfft2);
    fftw_cleanup();
}
