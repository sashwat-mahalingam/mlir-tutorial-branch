#include <stdio.h>
#include <time.h>

#define M 600
#define N 600
#define P 600   
float A[M][N][P];
float B[M][N][P];
float W[M][N][P];

void init_array()
{
    int i, j, k;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            for (k = 0; k < P; k++) {
                A[i][j][k] = (1+(i*j*k)%1024)/2.0;
                B[i][j][k] = 0;
            }
        }
    }

    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 3; k++) {
                W[i][j][k] = (i+j-2*k);
            }
        }
    }


}

void kernel() {
    int i, j, k;
    for (i = 1; i < M - 1; i++)
	    for (j = 1; j < N - 1; j++)
	        for (k = 1; k < P - 1; k++)
                B[i][j][k] = A[i][j][k] * W[1][1][1] + A[i-1][j][k] * W[0][1][1] + A[i+1][j][k] * W[2][1][1] + A[i][j-1][k] * W[1][0][1] + A[i][j+1][k]   * W[1][2][1] + A[i][j][k-1] * W[1][1][0] + A[i][j][k+1]   * W[1][1][2];            
}

int main()
{
    int i, j, k;

    init_array();
    clock_t start = clock();

    kernel();
    clock_t end = clock();
    double time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("%f\n", time_taken);
}