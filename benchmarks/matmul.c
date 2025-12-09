#include <stdio.h>
#include <time.h>

#define M 1024
#define N 1024
#define P 1024
float A[M][N];
float B[N][P];
float C[M][P];

void init_array()
{
    int i, j;

    for (i = 0; i < M; i++) {
        for (j = 0; j < N; j++) {
            A[i][j] = (1+(i*j)%1024)/2.0;
        }
    }

    for (i = 0; i < N; i++) {
        for (j = 0; j < P; j++) {
            B[i][j] = (1+(i*j)%1024)/2.0;
        }
    }

    for (i = 0; i < M; i++) {
        for (j = 0; j < P; j++) {
            C[i][j] = 0.0;
        }
    }
}

void kernel() {
    int i, j, k;
    for (i = 0; i < M; i++) {
        for (j = 0; j < P; j++) {
            for (k = 0; k < N; k++)
                C[i][j] = C[i][j] + A[i][k] * B[k][j];
        }
    }
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
