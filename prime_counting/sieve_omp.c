/* sieve_omp.c - OpenMP segmented sieve
   Compile: gcc -O2 -fopenmp sieve_omp.c -o sieve_omp -lm
   Run    : ./sieve_omp 4        (4 = number of threads) */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

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
    int threads = atoi(argv[1]);
    omp_set_num_threads(threads);

    double start = omp_get_wtime();

    find_small_primes();
    int num_blocks = (N - 1 + BLOCK - 1) / BLOCK;
    int count = 0;

    #pragma omp parallel reduction(+:count)
    {
        char buf[BLOCK];                       /* declared inside = every thread has its own */
        #pragma omp for schedule(dynamic)
        for (int b = 0; b < num_blocks; b++)
            count += sieve_block(b, buf);
    }

    printf("Threads = %d\n", threads);
    printf("Primes up to N = %d (expected 455052511 for 10^10)\n", count);
    printf("Time = %.3f s\n", omp_get_wtime() - start);
    return 0;
}
