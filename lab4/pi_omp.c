/**
 * INF560 - TD4
 *
 * Part 2-a
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main(int argc, char **argv)
{
    int N;
    int i;
    int seed;
    int m = 0;
    double pi;
    double time_start, time_stop, duration;

    double x, y;
    struct drand48_data drand_buf;

    /* Check the number of command-line arguments */
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return EXIT_FAILURE;
    }

    /* Grab the input parameters from command line */
    N = atoi(argv[1]);

    /* Check input-parameter values */
    if (N <= 0)
    {
        fprintf(stderr, "Error: N should be positive\n");
        return EXIT_FAILURE;
    }

    /* Start timer */
    time_start = omp_get_wtime();

#pragma omp parallel private(i, x, y, seed, drand_buf) shared(N)
    {
        seed = omp_get_thread_num();
        srand48_r(seed, &drand_buf);

        for (i = 0; i < N; i++)
        {
            drand48_r(&drand_buf, &x);
            drand48_r(&drand_buf, &y);

            if ((x * x + y * y) <= 1.0)
            {
#if DEBUG
                printf("x=%lf, y=%f is IN\n", x, y);
#endif
                m++;
            }
            else
            {
#if DEBUG
                printf("x=%lf, y=%f is OUT\n", x, y);
#endif
            }
        }
    }

    /* Stop timer */
    time_stop = omp_get_wtime();

#if DEBUG
    printf("m=%d\n", m);
#endif

    /* Compute value of PI */
    pi = (double)4 * m / N;

    printf("Result -> PI = %f\n", pi);

    /* Compute final duration (in seconds) */
    duration = time_stop - time_start;

    printf("Computed in %g s\n", duration);

    return EXIT_SUCCESS;
}