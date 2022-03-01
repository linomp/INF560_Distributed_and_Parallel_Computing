/**
 * INF560 - TD4
 *
 * Part 2-b
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int
main( int argc, char ** argv )
{
	int N ;
	int seed ;
	int i ;
	int m = 0 ;
	double pi ;
	double time_start, time_stop, duration ;
	int n_threads ;

	/* Check the number of command-line arguments */
	if ( argc != 3 )
	{
		fprintf( stderr, "Usage: %s N seed\n", argv[0]);
		return EXIT_FAILURE;
	}

	/* Grab the input parameters from command line */
	N = atoi(argv[1]);
	seed = atoi(argv[2]);

	/* Check input-parameter values */
	if ( N <= 0 ) 
	{
		fprintf( stderr, "Error: N should be positive\n" ) ;
		return EXIT_FAILURE ;
	}

#pragma omp parallel
	{
#pragma omp master
		{
			n_threads = omp_get_num_threads() ;
		}
	}

	printf( "Running on %d thread(s) w/ N=%d, seed=%d\n", 
			n_threads, N, seed ) ;

	/* Star timer */
	time_start = omp_get_wtime() ;

#pragma omp parallel default(none) shared(N,m) firstprivate(seed) private(i)
	{
		struct drand48_data data ;

		/* Shift the seed to get a different sequence of numbers
		 * per thread */
		seed += omp_get_thread_num() ;

		/* Initialize the pseudo-random number generator */
		srand48_r(seed, &data);

#pragma omp for reduction(+: m)
		for( i = 0 ; i < N ; i++ ) 
		{
			double x, y;

			drand48_r(&data, &x);
			drand48_r(&data, &y);

			x = 1 - (2*x) ;
			y = 1 - (2*y) ;

			if((x*x + y*y) < 1) 
			{
#if DEBUG
				printf("x=%lf, y=%f is IN\n", x, y);
#endif
				m++;
			} else {
#if DEBUG
				printf("x=%lf, y=%f is OUT\n", x, y);
#endif
			}
		}
	}

	/* Stop timer */
	time_stop = omp_get_wtime() ;

#if DEBUG
	printf("m=%d\n", m);
#endif

	/* Compute value of PI */
	pi = (double)4*m/N;

	printf("Result -> PI = %f\n", pi);

	/* Compute final duration (in seconds) */
	duration = time_stop - time_start ;

	printf("Computed in %g s\n", duration);

	return EXIT_SUCCESS ;
}
