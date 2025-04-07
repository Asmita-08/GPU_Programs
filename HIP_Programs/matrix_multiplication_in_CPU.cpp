#include <stdio.h>

#define N 2

int main() {
    int A[N][N] = {{1, 2}, {3, 4}};
    int B[N][N] = {{5, 6}, {7, 8}};
    int C[N][N] = {0};

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    printf("Result:\n");
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            printf("%d%c", C[i][j], (j == N - 1) ? '\n' : ' ');

    return 0;

}
