#include "solver.h"

#define FOUR_PI_SQUARED 39.478417604357432


double hh = 1.0 / (N_DISCR*N_DISCR);
int nRealElem = N_DISCR*N_DISCR;
int nCplxElem = N_DISCR*(N_DISCR/2+1);


/*
 *  Compute one iteration using BDF2 & AB2
 *  Return value is done in-place.
 */
double *k;
fftw_complex *c_hat;

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
fftw_complex *tmp;
fftw_complex *buffer;
fftw_complex *f_hat_0, *f_hat_1;
fftw_complex *c_hat_0, *c_hat_1;

void imex(double dt) {
    // Save current iteration
    tmp = c_hat_0;
    c_hat_0 = c_hat;
    c_hat = tmp;

    // Compute ĉ³ - c
    non_linear_term(c_hat_0);
    memcpy(f_hat_0, cval, nCplxElem*sizeof(fftw_complex));

    // Compute ĉ_{i+1}
    if (iter == 1) {    // IMEX-BDF1
        for(int i = 0; i < nCplxElem; i++) {
            c_hat[i] = (c_hat_0[i] + dt*f_hat_0[i]) / (1.0 + dt*1e-4*k[i]*k[i]);
        }

    } else {            // IMEX-BDF2
        for(int i = 0; i < nCplxElem; i++) {
            c_hat[i] = (4.0*c_hat_0[i] - c_hat_1[i] + 2.0*dt*(2.0*f_hat_0[i] - f_hat_1[i])) / (3.0 + 2e-4*dt*k[i]*k[i]);
        }
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

/*
 *  ETDRK solver.
 */
double *e1, *e2, *f1, *f2, *f3, *q;
fftw_complex *fa, *fb, *fc, *Nu, *Na, *Nb;

void etdrk4(double dt) {
    // Compute N(u)
    non_linear_term(c_hat);
    memcpy(Nu, cval, nCplxElem*sizeof(fftw_complex));

    // Compute a & N(a)
    for (int i = 0; i < nCplxElem; i++) {
        fa[i] = e2[i]*c_hat[i] + q[i]*Nu[i];
    }
    non_linear_term(fa);
    memcpy(Na, cval, nCplxElem*sizeof(fftw_complex));

    // Compute b & N(b)
    for (int i = 0; i < nCplxElem; i++) {
        fb[i] = e2[i]*c_hat[i] + q[i]*Na[i];
    }
    non_linear_term(fb);
    memcpy(Nb, cval, nCplxElem*sizeof(fftw_complex));

    // Compute c & N(b)
    for (int i = 0; i < nCplxElem; i++) {
        fc[i] = e2[i]*fa[i] + q[i]*(2.0*Nb[i] - Nu[i]);
    }
    non_linear_term(fc);

    // Compute u_{i+1}
    for (int i = 0; i < nCplxElem; i++) {
        c_hat[i] = e1[i]*c_hat[i] + f1[i]*Nu[i] + 2.0*f2[i]*(Na[i] + Nb[i]) + f3[i]*cval[i];
    }
}

/*
 *  Compute -k*F(c³ -c) where F is the Fourier transform.
 *  Result is stored in the "rval" array.
 */
void non_linear_term(fftw_complex *c) {
    for(int i = 0; i < nCplxElem; i++) {
        cval[i] = hh*c[i];
    }
    fftw_execute(irfft2);

    for (int i = 0; i < nRealElem; i++) {
        rval[i] = rval[i] - rval[i]*rval[i]*rval[i];
    }
    fftw_execute(rfft2);

    for(int i = 0; i < nCplxElem; i++) {
        cval[i] *= k[i];
    }
}

/*
 *  Initialise the various stuff
 */
void init_solver(double *c, double dt) {
    // Wavenumber k
    k = (double*) malloc(nCplxElem*sizeof(double));
    for (int i = 0; i < N_DISCR; i++) {
        for (int j = 0; j < N_DISCR/2+1; j++) {
            int l = (i < N_DISCR/2) ? i : i-N_DISCR;
            k[i*(N_DISCR/2+1)+j] = FOUR_PI_SQUARED * (j*j + l*l);
        }
    }

    // Initialise solver-specific buffers
    switch (SOLVER) {
        case IMEX:
            buffer  = (fftw_complex*) malloc(5*nCplxElem*sizeof(fftw_complex));
            c_hat   = &buffer[0];
            c_hat_0 = &buffer[  nCplxElem];
            c_hat_1 = &buffer[2*nCplxElem];
            f_hat_0 = &buffer[3*nCplxElem];
            f_hat_1 = &buffer[4*nCplxElem];
            break;

        case ETDRK4:
            fa    = (fftw_complex*) malloc(7*nCplxElem*sizeof(fftw_complex));
            fb    = &fa[  nCplxElem];
            fc    = &fa[2*nCplxElem];
            Nu    = &fa[3*nCplxElem];
            Na    = &fa[4*nCplxElem];
            Nb    = &fa[5*nCplxElem];
            c_hat = &fa[6*nCplxElem];

            e1 = (double*) malloc(6*nCplxElem*sizeof(double));
            e2 = &e1[  nCplxElem];
            f1 = &e1[2*nCplxElem];
            f2 = &e1[3*nCplxElem];
            f3 = &e1[4*nCplxElem];
            q  = &e1[5*nCplxElem];

            for (int i = 0; i < nCplxElem; i++) {
                double l = - 1e-4*k[i]*k[i];
                e1[i] = exp(l*dt);
                e2[i] = exp(l*dt / 2.0);

                int m = 32;
                complex q_tmp = 0.0, f1_tmp = 0.0, f2_tmp = 0.0, f3_tmp = 0.0;
                for (int j = 0; j < m; j++){
                    complex r = l*dt + cexp(I*M_PI * (j+.5)/m);

                    q_tmp  += (cexp(r/2.0) - 1.0) / r;
                    f1_tmp += (-4.0 -     r       + cexp(r)*( 4.0 - 3.0*r + r*r)) / (r*r*r);
                    f2_tmp += ( 2.0 +     r       + cexp(r)*(-2.0 + r))           / (r*r*r);
                    f3_tmp += (-4.0 - 3.0*r - r*r + cexp(r)*( 4.0 - r))           / (r*r*r);
                }
                q[i]  = dt*creal( q_tmp)/m;
                f1[i] = dt*creal(f1_tmp)/m;
                f2[i] = dt*creal(f2_tmp)/m;
                f3[i] = dt*creal(f3_tmp)/m;
            }
            break;
    }

    // FFTW
    cval = fftw_alloc_complex(nCplxElem);
    rval = fftw_alloc_real(nRealElem);
    rfft2  = fftw_plan_dft_r2c_2d(N_DISCR, N_DISCR, rval, cval, FFTW_PATIENT);
    irfft2 = fftw_plan_dft_c2r_2d(N_DISCR, N_DISCR, cval, rval, FFTW_PATIENT);

    // Initialise c_hat
    memcpy(rval, c, nRealElem*sizeof(double));
    fftw_execute(rfft2);
    memcpy(c_hat, cval, nCplxElem*sizeof(fftw_complex));
}

/*
 *  Free the various allocated arrays
 */
void free_solver() {
    // Solver
    switch (SOLVER) {
        case IMEX:
            free(buffer);
            break;

        case ETDRK4:
            free(fa);
            free(e1);
            break;
    }

    // Wavenumber
    free(k);

    // FFTW
    fftw_free(cval);
    fftw_free(rval);
    fftw_destroy_plan(rfft2);
    fftw_destroy_plan(irfft2);
    fftw_cleanup();
}

/*
 *  Convert solution back to real domain for display
 */
void getSolution(double *c) {
    memcpy(cval, c_hat, nCplxElem*sizeof(fftw_complex));
    fftw_execute(irfft2);
    for(int i = 0; i < nRealElem; i++) {
        c[i] = hh*rval[i];
    }
}
