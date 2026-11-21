/* sieve_mpi.c - MPI segmented sieve
   Rank r sieves blocks r, r+size, r+2*size, ... (blocks are dealt out like cards)
   Compile: mpicc -O2 sieve_mpi.c -o sieve_mpi -lm
   Run    : mpiexec -n 4 ./sieve_mpi */
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define N 10000000000LL      /* 10^10. Needs long long because it is bigger than int */
#define BLOCK 100000         /* numbers per block (100 KB, small enough for cache) */

int small_primes[10000];     /* primes up to sqrt(N) = 100000 (there are only 9592) */
int num_small = 0;

/* Plain sieve: find all primes up to sqrt(N) */
void find_small_primes() {
    int r = (int)sqrt((double)N);
    static char composite[100002];
    for (int i = 2; i <= r; i++) {
        if (composite[i] == 0) {
            small_primes[num_small++] = i;
            for (int j = 2 * i; j <= r; j += i)
                composite[j] = 1;
        }
    }
}

/* Count primes in block number b. Block b covers [2 + b*BLOCK, 2 + (b+1)*BLOCK - 1]. */
int sieve_block(int b, char *buf) {
    long long low  = 2 + (long long)b * BLOCK;
    long long high = low + BLOCK - 1;
    if (high > N) high = N;
    int len = (int)(high - low + 1);

    for (int i = 0; i < len; i++) buf[i] = 1;          /* 1 = prime so far */

    for (int k = 0; k < num_small; k++) {
        long long p = small_primes[k];
        if (p * p > high) break;
        long long start = (low + p - 1) / p * p;       /* first multiple of p in the block */
        if (start < p * p) start = p * p;
        for (long long j = start; j <= high; j += p)
            buf[j - low] = 0;                          /* cross out multiple */
    }

    int count = 0;
    for (int i = 0; i < len; i++) count += buf[i];
    return count;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    find_small_primes();                       /* every rank builds its own small primes */
    int num_blocks = (N - 1 + BLOCK - 1) / BLOCK;
    char buf[BLOCK];
    int local = 0, total = 0;

    for (int b = rank; b < num_blocks; b += size)
        local += sieve_block(b, buf);

    MPI_Reduce(&local, &total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Processes = %d\n", size);
        printf("Primes up to N = %d (expected 455052511 for 10^10)\n", total);
        printf("Time = %.3f s\n", end - start);
    }
    MPI_Finalize();
    return 0;
}
