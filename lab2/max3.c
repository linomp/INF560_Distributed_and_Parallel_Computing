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
  int e = 0, i = 0, j = 0;

  int *max;
  int **arrays;

  int *partials; // extra array to hold partial results
  int t;
  int temp;
  MPI_Status status;

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

  partials = malloc(sizeof(int) * size);
  if (partials == NULL)
  {
    fprintf(stderr, "Unable to allocate %d elements for partials array\n", n);
    return 1;
  }

  /* start the measurement */
  t1 = MPI_Wtime();

  /* search for the max value in the assigned array chunk */
  long chunk_size = n / size;
  long chunk_init = rank * chunk_size;

  for (i = 0; i < m; i++)
  {
    max[i] = arrays[i][chunk_init];
    for (j = chunk_init; j < (chunk_init + chunk_size); j++)
    {
      if (arrays[i][j] > max[i])
      {
        max[i] = arrays[i][j];
      }
    }

    if (rank == 0)
    {
      partials[0] = max[i];
      for (t = 1; t < size; t++)
      {
        MPI_Recv(&temp, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        partials[t] = temp;
#if DEBUG
        printf("Message from rank %d: %d\n", status.MPI_SOURCE, temp);
#endif
      }
      /* search for the max among partial results of ith-array */
      max[i] = partials[0];
      for (e = 0; e < size; e++)
      {
        if (partials[e] > max[i])
        {
          max[i] = partials[e];
        }
      }
    }
    else
    {
      MPI_Send(&(max[i]), 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
    }
  }

  /* stop the measurement */
  t2 = MPI_Wtime();

  printf("Computation time: %f s\n", t2 - t1);

#if DEBUG
  if (rank == 0)
  {
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
  }
#endif

  if (rank == 0)
  {
    printf("(Seed: %d, Size: %d, # Arrays: %d) Time = %g s, \nMax values =\n", s, n, m, t2 - t1);
    for (i = 0; i < m; i++)
    {
      printf("%d\n", max[i]);
    }
  }
  MPI_Finalize();
  return 0;
}
