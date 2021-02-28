#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "./matplotlib-cpp-master/matplotlibcpp.h"
#include <omp.h>


namespace plt = matplotlibcpp;

#ifndef max
#define max(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef min
#define min(a,b)            (((a) < (b)) ? (a) : (b))
#endif

int parity(int di, int dj, int i, int j, int rho);

int main(int argc, char *argv[]){

	int nx = 512;
	int ny = 512;
	double h = 1.0/nx ;

	// memory allocation
	float* u = (float*)calloc(nx*ny, sizeof(float));
	//std::vector<float> z(nx * ny);

	//writing file init
    FILE *fp = fopen("viscous.txt","w+");

	//init
	for(int i=0; i<nx*ny; i++){
		//Double circle
		// if((((i/nx - 150)*(i/nx - 150) + (i%nx -170)*(i%nx - 170) < 75*75) && i/nx > 175) || (((i/nx - 150)*(i/nx - 150) + (i%nx -342)*(i%nx - 342) < 75*75) && i/nx > 175)){
		// 	u[i] = 0.0;
		// }
		// else{
		// 	u[i] = 0.005;
		// }
		//CENTERED CIRCLE
		// if((i/nx - 256)*(i/nx - 256) + (i%nx -256)*(i%nx - 256) < 75*75){
		// 	u[i] = 0.03;
		// }
		// else{
		// 	u[i] = 0.01;
		// }
		u[i] = 0.005;
	}

	//BIG LINE
	// for(int index=350*nx ; index<390*nx ; index++){
	// 	u[index] = 0.1;
	// }

	// SIMPLE Gaussian
	// double mu_x[1] = {0.5};
	// double mu_y[1] = {0.6};
	// double sigma_x[1] = {0.1};
	// double sigma_y[1] = {0.07};
	// double density, x, y;
	// int i, j;
	// double max = 0;

	// for(int l=0; l<sizeof(mu_x); l++){
	// 	for(int index=0 ; index<nx*ny ; index++){
	// 		i = (int) index % nx;
	// 		j = (int) index / nx;

	// 		x = i*h;
	// 		y = j*h;

	// 		density = (1.0/(600.0*2.0*M_PI*sigma_x[l]*sigma_y[l])) * exp(-(1.0/2.0)*((x-mu_x[l])*(x-mu_x[l])/(sigma_x[l]*sigma_x[l]) + (y-mu_y[l])*(y-mu_y[l])/(sigma_y[l]*sigma_y[l])));
	// 		if (density > u[index]){
	// 			u[index] = density;
	// 			if (density > max){
	// 				max = density;
	// 			}
	// 		}
	// 	}
	// }
	// printf("Max = %f \n", max);

	//Gaussian
	double mu_x[5] = {0.18, 0.2, 0.56, 0.6, 0.9};
	double mu_y[5] = {0.8, 0.45, 0.7, 0.3, 0.5};
	double sigma_x[5] = {0.1, 0.1, 0.1, 0.1, 0.1};
	double sigma_y[5] = {0.07, 0.07, 0.07, 0.07, 0.07};
	double density, x, y;
	int i, j;
	double max = 0;

	for(int l=0; l<sizeof(mu_x); l++){
		for(int index=0 ; index<nx*ny ; index++){
			i = (int) index % nx;
			j = (int) index / nx;

			x = i*h;
			y = j*h;

			density = (1.0/(500.0*2.0*M_PI*sigma_x[l]*sigma_y[l])) * exp(-(1.0/2.0)*((x-mu_x[l])*(x-mu_x[l])/(sigma_x[l]*sigma_x[l]) + (y-mu_y[l])*(y-mu_y[l])/(sigma_y[l]*sigma_y[l])));
			if (density > u[index]){
				u[index] = density;
				if (density > max){
					max = density;
				}
			}
		}
	}
	printf("Max = %f \n", max);

	//Merging gaussian
	// double mu_x[2] = {0.5, 0.65};
	// double mu_y[2] = {0.6, 0.8};
	// double sigma_x[2] = {0.1, 0.1};
	// double sigma_y[2] = {0.1, 0.1};
	// double density, x, y;
	// int i, j;
	// double max = 0;

	// for(int l=0; l<sizeof(mu_x); l++){
	// 	for(int index=0 ; index<nx*ny ; index++){
	// 		i = (int) index % nx;
	// 		j = (int) index / nx;

	// 		x = i*h;
	// 		y = j*h;

	// 		density = (1.0/(500.0*2.0*M_PI*sigma_x[l]*sigma_y[l])) * exp(-(1.0/2.0)*((x-mu_x[l])*(x-mu_x[l])/(sigma_x[l]*sigma_x[l]) + (y-mu_y[l])*(y-mu_y[l])/(sigma_y[l]*sigma_y[l])));
	// 		if (density > u[index]){
	// 			u[index] = density;
	// 			if (density > max){
	// 				max = density;
	// 			}
	// 		}
	// 	}
	// }
	// printf("Max = %f \n", max);

	//BORDER
	for(int i=0; i<nx; i++){
		u[i] = 0.;
		u[nx*(ny-1) + i] = 0.;
	}

	for(int j=0; j<ny; j++){
		u[nx*j] = 0.;
		u[nx*j + nx-1] = 0.;
	}

	//WRITE IN THE FILE
	for (int i=0; i<nx*ny; i++){
		fprintf(fp, "%f, ", u[i]);
	}
	fprintf(fp, "\n");

	double tau = 0.01 ;
	double e = 5.0;
	double eta = 2.0;
	double G = 10;
	double beta = 0.01;
	int n_passe = 10;
	char title[50];

	omp_set_num_threads(6);

	//LOOP IN TIME
	for(int t = 0; t < 100; t++){
		for(int p=0; p<n_passe; p++){

			//Flux in direcion (di, dj) = (1,0) Horizontal
			int di = 1;
			int dj = 0;
			// for(int m=0; m<nx*ny; m++){
			// 	if((((m/nx - 150)*(m/nx - 150) + (m%nx -170)*(m%nx - 170) < 75*75) && m/nx > 175) || (((m/nx - 150)*(m/nx - 150) + (m%nx -342)*(m%nx - 342) < 75*75) && m/nx > 175)){
			// 		u[m] = 0.0;
			// 	}
			// }


			for(int rho=0; rho<4; rho++){
				#pragma omp parallel for
				for(int k=2; k<nx*ny; k++){
					int rho_ij, i_p, j_p;
					double W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
					double a,b;
					int i,j;

					i = (int) k % nx;
					j = (int) k / nx;

					if(i==0 || i==1 || i==nx-1 || i==nx-2 || j==0 || j==1 || j==ny-1 || j==ny-2){
						//do nothing
					}
					else{
						rho_ij = parity(di, dj, i, j, rho);
						if (rho_ij == 0 || rho_ij == 2){
							i_p = i + di;
							j_p = j + dj;
						} else{
							i_p = i - di;
							j_p = j - dj;
						}

						lap_q = (u[nx*j_p + (i_p+1)] + u[nx*(j_p+1) + i_p] + u[nx*(j_p-1) + i_p]);
						lap_p = (u[nx*j + (i-1)] + u[nx*(j+1) + i] + u[nx*(j-1) + i]);

						W_q = G*(ny-j_p-0.5);
						W_p = G*(ny-j-0.5);

						//M = (2.0/3.0) * 1.0/(1.0/(u[nx*j_p + i_p]*u[nx*j_p + i_p]*u[nx*j_p + i_p]) + 1.0/(u[nx*j + i]*u[nx*j + i]*u[nx*j + i]));
						M = 2.0*u[nx*j_p + i_p]*u[nx*j_p + i_p]*u[nx*j + i]*u[nx*j + i]/(3.0*(u[nx*j_p + i_p] + u[nx*j + i])); //+ (beta/2.0)*(u[nx*j_p + i_p]*u[nx*j_p + i_p] + u[nx*j + i]*u[nx*j + i]);

						theta = 1.0 + (2.0*tau*M*(5.0*e + eta));

						f = -(M/(theta)) * ((5.0*e+eta)*(u[nx*j_p + i_p] - u[nx*j + i]) - e*(lap_q - lap_p) + W_q-W_p);
						// if(f>0.001 || f<-0.001){
						// 	printf("HORIZONTAL: anisotropic: %f, surface tension: %f, gravity: %f \n", eta*(u[nx*j_p + i_p] - u[nx*j + i]) ,- e*(lap_q - lap_p), W_q-W_p);
						// }

						a = -u[nx*j_p + i_p];
						b = u[nx*j + i];

						delta_u = max(a, min(tau*f/h, b));

						if(rho_ij == 3){
							u[nx*j + i] = u[nx*j + i] - delta_u;
							u[nx*j_p + i_p] = u[nx*j_p + i_p] + delta_u;
						}
					}
				}
			}

			//Flux in direcion (di, dj) = (0,1) Vertical
			di = 0;
			dj = 1;

			for(int rho=0; rho<4; rho++){
				#pragma omp parallel for
				for(int k=2; k<nx*ny; k++){
					int rho_ij, i_p, j_p;
					double W_q, W_p, M, theta, f, delta_u, lap_p, lap_q;
					double a,b;
					int i,j;

					i = (int) k % nx;
					j = (int) k / nx;

					if(i==0 || i==1 || i==nx-1 || i==nx-2 || j==0 || j==1 || j==ny-1 || j==ny-2){
						//do nothing
					}
					else{
						rho_ij = parity(di, dj, i, j, rho);
						if (rho_ij == 0 || rho_ij == 2){
							i_p = i + di;
							j_p = j + dj;
						} else{
							i_p = i - di;
							j_p = j - dj;
						}

						lap_q = (u[nx*j_p + (i_p+1)] + u[nx*j_p + (i_p-1)] + u[nx*(j_p+1) + i_p]);
						lap_p = (u[nx*j + (i+1)] + u[nx*j + (i-1)] + u[nx*(j-1) + i]);

						W_q = G*(ny-j_p-0.5);
						W_p = G*(ny-j-0.5);

						//M = (2.0/3.0) * 1.0/(1.0/(u[nx*j_p + i_p]*u[nx*j_p + i_p]*u[nx*j_p + i_p]) + 1.0/(u[nx*j + i]*u[nx*j + i]*u[nx*j + i]));
						M = 2.0*u[nx*j_p + i_p]*u[nx*j_p + i_p]*u[nx*j + i]*u[nx*j + i]/(3.0*(u[nx*j_p + i_p] + u[nx*j + i])); //+ (beta/2.0)*(u[nx*j_p + i_p]*u[nx*j_p + i_p] + u[nx*j + i]*u[nx*j + i]);

						theta = 1.0 + (2.0*tau*M*(5.0*e + eta));

						f = -(M/(theta)) * ((5.0*e + eta)*(u[nx*j_p + i_p] - u[nx*j + i]) - e*(lap_q - lap_p) + W_q-W_p);
						// if(f>0.0001 || f<-0.0001){
						// 	printf("VERTICAL: anisotropic: %f, surface tension: %f, gravity: %f \n", eta*(u[nx*j_p + i_p] - u[nx*j + i]) ,- e*(lap_q - lap_p), W_q-W_p);
					 // 	}

						a = -u[nx*j_p + i_p];
						b = u[nx*j + i];

						delta_u = max(a, min(tau*f/h, b));

						if(rho_ij == 3){
							u[nx*j + i] = u[nx*j + i] - delta_u;
							u[nx*j_p + i_p] = u[nx*j_p + i_p] + delta_u;
						}
					}

				}
			}
		}

    const float* zptr = &(u[0]);
    const int colors = 1;

		plt::clf();

		sprintf(title, "Time = %f", t*tau);
    plt::title(title);
    plt::imshow(zptr, nx, ny, colors);

    // Show plots
    plt::pause(0.0001);

		printf("t = %d\n", t);





		//printf("time = %d \n",t);
		//if(t==50 || t==100 || t==150 || t==200){ //t==50 || t==100 || t==150 || t==200 || t==250 || t==300 || t==350 || t==400
			//for (int i=0; i<nx*ny; i++){
				//fprintf(fp, "%f, ", u[i]);
			//}
			//fprintf(fp, "\n");
		//}


	}

	//free memory
	free(u);

	fclose(fp);

	printf("\n *Happy computer sound* \n");
	return 0;
}


int parity(int di, int dj, int i, int j, int rho){
	return ((dj+1)*i + (di+1)*j + rho) % 4;
}
