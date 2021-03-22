#include "solver.h"

#define REAL 0
#define CPLX 1
#define FOUR_PI_SQUARED 39.478417604357432


double hh = 1.0 / (N_DISCR*N_DISCR);
int nRealElem = N_DISCR*N_DISCR;
int nCplxElem = N_DISCR*(N_DISCR/2+1);


/*
 *  Compute one iteration using BDF2 & AB2
 *  Return value is done in-place.
 */
int iter = 1;
void *tmp;
double *k;
fftw_complex *buffer;
fftw_complex *f_hat, *f_hat_1;
fftw_complex *c_hat, *c_hat_1;

void step(double *c, double dt) {
    // Compute ĉ
    memcpy(rval, c, nRealElem*sizeof(double));
    fftw_execute(rfft2);
    memcpy(c_hat, cval, nCplxElem*sizeof(fftw_complex));

    // Compute ĉ³ - c
    for(int i = 0; i < nRealElem; i++) {
        rval[i] = c[i]*c[i]*c[i] - c[i];
    }
    fftw_execute(rfft2);
    memcpy(f_hat, cval, nCplxElem*sizeof(fftw_complex));

    // Compute ĉ_{i+1}
    if (iter == 1) {    // IMEX-BDF1
        for(int i = 0; i < nCplxElem; i++) {
            cval[i][REAL] = hh * (c_hat[i][REAL] - dt*k[i]*f_hat[i][REAL]) / (1.0 + dt*1e-4*k[i]*k[i]);
            cval[i][CPLX] = hh * (c_hat[i][CPLX] - dt*k[i]*f_hat[i][CPLX]) / (1.0 + dt*1e-4*k[i]*k[i]);
        }

    } else {            // IMEX-BDF2
        for(int i = 0; i < nCplxElem; i++) {
            cval[i][REAL] = hh * (4.0*c_hat[i][REAL] - c_hat_1[i][REAL] - 2.0*dt*k[i] * (2.0*f_hat[i][REAL] - f_hat_1[i][REAL])) / (3.0 + 2e-4*dt*k[i]*k[i]);
            cval[i][CPLX] = hh * (4.0*c_hat[i][CPLX] - c_hat_1[i][CPLX] - 2.0*dt*k[i] * (2.0*f_hat[i][CPLX] - f_hat_1[i][CPLX])) / (3.0 + 2e-4*dt*k[i]*k[i]);
        }
    }

    // Back to physical domain
    fftw_execute(irfft2);
    memcpy(c, rval, nRealElem*sizeof(double));

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
void init_solver(double *c) {
    // Semi-implicit scheme
    buffer  = fftw_alloc_complex(4*nCplxElem);
    c_hat   = &buffer[0];
    c_hat_1 = &buffer[  nCplxElem];
    f_hat   = &buffer[2*nCplxElem];
    f_hat_1 = &buffer[3*nCplxElem];

    // Wavenumber k
    k = (double*) malloc(nCplxElem*sizeof(double));
    for (int i = 0; i < N_DISCR; i++) {
        for (int j = 0; j < N_DISCR/2+1; j++) {
            int l = (i < N_DISCR/2) ? i : i-N_DISCR;
            k[i*(N_DISCR/2+1)+j] = FOUR_PI_SQUARED * (j*j + l*l);
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
    // Semi-implicit scheme
    fftw_free(buffer);
    free(k);

    // FFTW
    fftw_free(cval);
    fftw_free(rval);
    fftw_destroy_plan(rfft2);
    fftw_destroy_plan(irfft2);
    fftw_cleanup();
}
