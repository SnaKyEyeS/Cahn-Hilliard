#include "functions.h"
#include "kernel.h"

#define REAL 0
#define CPLX 1



/*
 *  Compute the 2-D, cartesian Laplacian of c
 *  Return value is not in-place.
 */
// void laplacian(double* c, double h, double* delsq){
//
//     // Forward 2D real-valued FFT
//     memcpy(rval, c, N_DISCR*N_DISCR*sizeof(double));
//     fftw_execute(rfft2);
//
//     // Take the derivative
//     int l, ind;
//     double k;
//     double factor = 4*M_PI*M_PI*h*h;
//     for (int i = 0; i < N_DISCR; i++) {
//         for (int j = 0; j < N_DISCR/2+1; j++) {
//
//             // Wavenumber
//             l = (i < N_DISCR/2) ? i : i-N_DISCR;
//             k = -factor * (j*j + l*l);
//
//             // Multiply by (ik)²
//             ind = i*(N_DISCR/2+1)+j;
//             cval[ind][REAL] = k*cval[ind][REAL];
//             cval[ind][CPLX] = k*cval[ind][CPLX];
//         }
//     }
//
//     // Backward 2D real-valued FFT
//     fftw_execute(irfft2);
//     memcpy(delsq, rval, N_DISCR*N_DISCR*sizeof(double));
// }

/*
 *  Initialise the various stuff
 */
// void init_functions() {
//
// #ifdef USE_OPENMP
//     omp_set_num_threads(6);
// #endif
//
//     k1    = (double*) malloc(N_DISCR*N_DISCR*sizeof(double));
//     k2    = (double*) malloc(N_DISCR*N_DISCR*sizeof(double));
//     k3    = (double*) malloc(N_DISCR*N_DISCR*sizeof(double));
//     k4    = (double*) malloc(N_DISCR*N_DISCR*sizeof(double));
//     tmp   = (double*) malloc(N_DISCR*N_DISCR*sizeof(double));
//     delsq = (double*) malloc(N_DISCR*N_DISCR*sizeof(double));
//
//     cval = fftw_alloc_complex(N_DISCR*(N_DISCR/2+1));
//     rval = fftw_alloc_real(N_DISCR*N_DISCR);
//
//     rfft2  = fftw_plan_dft_r2c_2d(N_DISCR, N_DISCR, rval, cval, FFTW_PATIENT);
//     irfft2 = fftw_plan_dft_c2r_2d(N_DISCR, N_DISCR, cval, rval, FFTW_PATIENT);
// }

/*
 *  Free the various allocated arrays
 */
void free_functions() {
    // RungeKutta4
    // free(k1);
    // free(k2);
    // free(k3);
    // free(k4);
    // free(tmp);
    //
    // // f
    // free(delsq);

    // laplacian
    // fftw_free(cval);
    // fftw_free(rval);
    // fftw_destroy_plan(rfft2);
    // fftw_destroy_plan(irfft2);
    // fftw_cleanup();
}
