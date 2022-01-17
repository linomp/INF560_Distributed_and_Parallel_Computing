/**
 * INF560 - TD2
 *
 * Part 2: Work Decomposition
 * Sequential Max
 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  int rank, size;
  int s;
  int n;
  int m; // number of arrays to generate
  int i = 0, j = 0;

  int *max;
  int **arrays;

  double t1, t2;

  /* MPI Initialization */
  MPI_Init(&argc, &argv);

  /* Get the rank of the current task and the number
   * of MPI processe
   */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Check the input arguments */
  if (argc < 4)
  {
    printf("Usage: %s S N M\n", argv[0]);
    printf("\tS: seed for pseudo-random generator\n");
    printf("\tN: size of the array\n");
    printf("\tM: number of arrays\n");
    exit(1);
  }

  s = atoi(argv[1]);
  n = atoi(argv[2]);
  m = atoi(argv[3]);
  srand48(s);

  /* Allocate max values and random num arrays */
  max = malloc(sizeof(int) * m);
  if (max == NULL)
  {
    fprintf(stderr, "Unable to allocate %d elements for max array \n", m);
    return 1;
  }

  arrays = (int **)malloc(sizeof(int *) * m);
  if (arrays == NULL)
  {
    fprintf(stderr, "Unable to allocate %d elements for data arrays \n", m);
    return 1;
  }

  for (i = 0; i < m; i++)
  {
    arrays[i] = (int *)malloc(sizeof(int) * n);
    if (arrays[i] == NULL)
    {
      fprintf(stderr, "Unable to allocate %d elements for array \n", n, i);
      return 1;
    }
  }

  /* Initialize the random num arrays */
  for (i = 0; i < m; i++)
  {
    for (j = 0; j < n; j++)
    {
      arrays[i][j] = lrand48() % n;
    }
  }

  /* start the measurement */
  t1 = MPI_Wtime();

  /* search for the max values */
  for (i = 0; i < m; i++)
  {
    max[i] = arrays[i][0];
    for (j = 0; j < n; j++)
    {
      if (arrays[i][j] > max[i])
      {
        max[i] = arrays[i][j];
      }
    }
  }

  /* stop the measurement */
  t2 = MPI_Wtime();

  printf("Computation time: %f s\n", t2 - t1);

#if DEBUG
  printf("the arrays contain:\n");
  for (i = 0; i < m; i++)
  {
    for (j = 0; j < n; j++)
    {
      printf("%d  ", arrays[i][j]);
    }
    printf("\n");
  }
  printf("\n");
#endif

  printf("(Seed: %d, Size: %d, # Arrays: %d) Time = %g s, \nMax values =\n", s, n, m, t2 - t1);
  for (i = 0; i < m; i++)
  {
    printf("%d\n", max[i]);
  }

  MPI_Finalize();
  return 0;
}
