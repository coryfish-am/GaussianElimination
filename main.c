#define _GNU_SOURCE

#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <fenv.h>
#include <setjmp.h>

#include "gauss_solve.h"
#include "helpers.h"

/* Size of the matrix */
#define N  3

// Function prototype for plu
void plu(int n, double A[n][n], int P[n]);

void test_plu() {
    printf("Entering function: %s\n", __func__);

    const double A0[N][N] = {
        {2, 3, -1},
        {4, 1, 2},
        {-2, 7, 2}
    };

    double A[N][N];
    int P[N]; // Permutation array

    memcpy(A, A0, sizeof(A0));

    // Call PLU decomposition
    plu(N, A, P);

    // Print the permutation vector P
    printf("Permutation Vector P:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", P[i]);
    }
    printf("\n");

    // Print U
    puts("U:\n");
    print_matrix(N, A, FLAG_UPPER_PART);

    // Print L
    puts("L:\n");
    print_matrix(N, A, FLAG_LOWER_PART);
}

void test_gauss_solve() {
    printf("Entering function: %s\n", __func__);
    
    const double A0[N][N] = {
        {2, 3, -1},
        {4, 1, 2},
        {-2, 7, 2}
    };

    const double b0[N] = {5, 6, 3};

    double A[N][N], b[N], x[N], y[N];

    /* Create copies of the matrices.
       NOTE: the copies will get destroyed. */
    memcpy(A, A0, sizeof(A0));
    memcpy(b, b0, sizeof(b0));  

    gauss_solve_in_place(N, A, b);

    memcpy(x, b, sizeof(b0));
    matrix_times_vector(N, A0, x, y);

    double eps = 1e-6, dist = norm_dist(N, b0, y);
    assert( dist < eps);

    /* Print x */
    puts("x:\n");
    print_vector(N, x);

    /* Print U */
    puts("U:\n");
    print_matrix(N, A, FLAG_UPPER_PART);
  
    /* Print L */
    puts("L:\n");
    print_matrix(N, A, FLAG_LOWER_PART);
}

jmp_buf env;  // Buffer to store the state for setjmp/longjmp

void test_gauss_solve_with_zero_pivot() {
    printf("Entering function: %s\n", __func__);
  
    double A[N][N] = {
        {0, 3, -1},
        {4, 1, 2},
        {-2, 7, 2}
    };

    double b[N] = {5, 6, 3};

    // Save the program state with setjmp
    if (setjmp(env) == 0) {
        gauss_solve_in_place(N, A, b);
        print_matrix(N, A, FLAG_LOWER_PART);
    } else {
        // This block is executed when longjmp is called
        printf("Returned to main program flow after exception\n");
    }
  
}

void test_lu_in_place() {
    printf("Entering function: %s\n", __func__);

    const double A0[N][N] = {
        {2, 3, -1},
        {4, 1, 2},
        {-2, 7, 2}
    };

    const double b0[N] = {5, 6, 3};

    double A[N][N];

    memcpy(A, A0, sizeof(A0));

    lu_in_place(N, A);


    /* Print U */
    puts("U:\n");
    print_matrix(N, A, FLAG_UPPER_PART);
  
    /* Print L */
    puts("L:\n");
    print_matrix(N, A, FLAG_LOWER_PART);

    lu_in_place_reconstruct(N, A);

    /* Print U */
    puts("Reconstructed A:\n");
    print_matrix(N, A, FLAG_WHOLE);

    memcpy(A, A0, sizeof(A0));
    puts("Original A:\n");
    print_matrix(N, A, FLAG_WHOLE);

    double eps = 1e-6;
    assert(frobenius_norm_dist(N, A, A0) < eps);
}

void benchmark_test(int n) {
    /* Allocate matrix on stack */
    double A0[n][n], A[n][n];
    generate_random_matrix(n, A0);
    copy_matrix(n, A0, A);

    lu_in_place(N, A);
    lu_in_place_reconstruct(N, A);

    double eps = 1e-6;
    assert(frobenius_norm_dist(N, A0, A) < eps);
}

void fpe_handler(int sig) {
    printf("Entering %s...\n", __func__);
    if(sig == SIGFPE) {
        printf("Floating point exception occurred, ignoring...\n");
        longjmp(env, 1);  // Jump back to where setjmp was called
    }
}

int main() {
    // Enable trapping for specific floating-point exceptions
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    sighandler_t old_handler = signal(SIGFPE, fpe_handler);

    test_gauss_solve();
    test_lu_in_place();
    test_plu();  // Adding the test for the PLU function
    benchmark_test(5);
    test_gauss_solve_with_zero_pivot();  
    exit(EXIT_SUCCESS);
}

