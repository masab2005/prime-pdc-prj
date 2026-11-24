/* factor_mpi.c - MPI prime factorization of 2..N
   Rank r takes the numbers 2+r, 2+r+size, 2+r+2*size, ... (numbers are dealt out like cards,
   so every rank gets a similar mix of easy and hard numbers)
   Compile: mpicc -O2 factor_mpi.c -o factor_mpi -lm
   Run    : mpiexec -n 4 ./factor_mpi */
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#define N 100000000          // factor every number from 2 to N (10^8). Works up to 10^9 without other changes

int primes[4000];            // small primes up to sqrt(N) (3401 primes are enough for N up to 10^9)
int num_primes = 0;

// Plain sieve: find all primes up to sqrt(N)
void find_small_primes() {
    int r = (int)sqrt((double)N);
    static char composite[32000];
    for (int i = 2; i <= r; i++) {
        if (composite[i] == 0) {
            primes[num_primes++] = i;
            for (int j = 2 * i; j <= r; j += i)
                composite[j] = 1;
        }
    }
}

// Break n into prime factors and return how many there are (12 = 2*2*3 gives 3)
int count_factors(int n) {
    int count = 0;
    for (int k = 0; k < num_primes && primes[k] * primes[k] <= n; k++) {
        while (n % primes[k] == 0) {
            n = n / primes[k];
            count++;
        }
    }
    if (n > 1) count++;          // what is left is one last prime
    return count;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    find_small_primes();                 // every rank builds its own small primes
    long long local = 0, total = 0;

    for (int n = 2 + rank; n <= N; n += size)
        local += count_factors(n);

    MPI_Reduce(&local, &total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Processes = %d\n", size);
        printf("Total prime factors = %lld (expected 392351272 for N = 10^8)\n", total);
        printf("Time = %.3f s\n", end - start);
    }
    MPI_Finalize();
    return 0;
}
