/* factor_omp.c - OpenMP prime factorization of 2..N
   Compile: gcc -O2 -fopenmp factor_omp.c -o factor_omp -lm
   Run    : ./factor_omp 4        (4 = number of threads) */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

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
    int threads = atoi(argv[1]);
    omp_set_num_threads(threads);

    double start = omp_get_wtime();

    find_small_primes();
    long long total = 0;

    // dynamic: some numbers take much longer than others, so free threads grab the next chunk of 10000
    #pragma omp parallel for schedule(dynamic, 10000) reduction(+:total)
    for (int n = 2; n <= N; n++)
        total += count_factors(n);

    printf("Threads = %d\n", threads);
    printf("Total prime factors = %lld (expected 392351272 for N = 10^8)\n", total);
    printf("Time = %.3f s\n", omp_get_wtime() - start);
    return 0;
}
