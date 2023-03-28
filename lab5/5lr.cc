#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int size;
    int rank;
    const int ROOT = 0;
    const int n = 16;

    int *a, *b, *c;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int cols_per = n / size;

    int local_counts[size], offsets[size];
    int sum = 0;
    for (int i = 0; i < size; i++) {
        local_counts[i] = n * n / size;
        offsets[i] = sum;
        sum += local_counts[i];
    }

    int* a_block = new int[cols_per * n];
    int* c_block = new int[cols_per * n];

    double time;
    if (rank == ROOT) {
        a = new int[n * n];
        b = new int[n * n];
        c = new int[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = i + j;
                b[i * n + j] = i - j;
                c[i * n + j] = 0;
            }
        }
        time = - MPI_Wtime();
    } else {
        b = new int[n * n];
    }

    MPI_Scatterv(a, local_counts, offsets, MPI_INT, a_block, cols_per * n, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(b, n * n, MPI_INT, ROOT, MPI_COMM_WORLD);

    for (int i = 0; i < cols_per; i++)
    {
        for (int j = 0; j < n; j++) {
            int sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum = sum + a_block[i * n + k] * b[k * n + j];
            }
            c_block[i * n + j] = sum;
        }
    }

    MPI_Gatherv(c_block, cols_per * n, MPI_INT, c, local_counts, offsets, MPI_INT, ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        time += MPI_Wtime();
        printf("Elapsed time: %f sec.\n", time);
    }

    MPI_Finalize();
    return 0;
}