#include "helper.h"


void init_etdrk4(double *e1, double *e2, double *f1, double *f2, double *f3, double *q, double dt) {
    for (int i = 0; i < N_DISCR; i++) {
        for (int j = 0; j < N_DISCR/2+1; j++) {
            // Index & wavenumber
            int ind = i*(N_DISCR/2+1)+j;
            int i_  = (i < N_DISCR/2) ? i : i-N_DISCR;

#ifdef CONSTANT_MOBILITY
            double k = FOUR_PI_SQUARED * (j*j + i_*i_);
            double l = - KAPPA*k*k;
#endif
#ifdef VARIABLE_MOBILITY
            double k = FOUR_PI_SQUARED * (j*j + i_*i_) / (N_DISCR*N_DISCR);
            double l = - .5*KAPPA*k*k;
#endif

            e1[ind] = exp(l*dt);
            e2[ind] = exp(l*dt / 2.0);

            int n_points = 32;
            complex q_tmp = 0.0, f1_tmp = 0.0, f2_tmp = 0.0, f3_tmp = 0.0;
            for (int n = 0; n < n_points; n++){
                complex r = l*dt + cexp(I*M_PI * (n+.5)/n_points);

                q_tmp  += (cexp(r/2.0) - 1.0) / r;
                f1_tmp += (-4.0 -     r       + cexp(r)*( 4.0 - 3.0*r + r*r)) / (r*r*r);
                f2_tmp += ( 2.0 +     r       + cexp(r)*(-2.0 + r))           / (r*r*r);
                f3_tmp += (-4.0 - 3.0*r - r*r + cexp(r)*( 4.0 - r))           / (r*r*r);
            }
            q[ind]  = dt*creal( q_tmp)/n_points;
            f1[ind] = dt*creal(f1_tmp)/n_points;
            f2[ind] = dt*creal(f2_tmp)/n_points;
            f3[ind] = dt*creal(f3_tmp)/n_points;
        }
    }
}
