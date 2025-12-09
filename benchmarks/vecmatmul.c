#include <stdio.h>
#include <time.h>

#define N 15625
float A[N];
float B[N];
float C[N][N];
float D[N][N];
void init_array()
{
    int i, j;
    for (i = 0; i < N; i++) {
        A[i] = (1+(i)%1024)/2.0;
        B[i] = (1+(N-i)%1024)/2.0;
    }

    for(i =0; i < N; i++) {
        for(j =0; j < N; j++) {
            C[i][j] = (1+i*j)%1024/2.0;
            D[i][j] = 0;
        }
    }

}

void kernel() {
    int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            D[i][j] = C[i][j] + A[i] * B[j];
        }
    }
}
int main()
{
    int i, j;

    init_array();
    clock_t start = clock();

    kernel();
    clock_t end = clock();
    double time_taken = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("%f\n", time_taken);
}