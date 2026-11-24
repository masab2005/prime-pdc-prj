/* factor_seq.c - sequential prime factorization of 2..N
   Compile: gcc -O2 -fopenmp factor_seq.c -o factor_seq -lm
   Run    : ./factor_seq */
#include <stdio.h>
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

int main() {
    double start = omp_get_wtime();

    find_small_primes();
    long long total = 0;                 // long long: the total can pass 2 billion

    for (int n = 2; n <= N; n++)
        total += count_factors(n);

    printf("Total prime factors = %lld (expected 392351272 for N = 10^8)\n", total);
    printf("Time = %.3f s\n", omp_get_wtime() - start);
    return 0;
}
