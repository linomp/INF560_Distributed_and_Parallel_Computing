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

  int t;
  int *array;
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

  /* start the measurement */
  t1 = MPI_Wtime();

  if (rank == 0)
  {
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

    if (m <= size)
    {
      // Round-Robin:
      // wrap around the array idx if number of processes > arrays
      for (i = 0; i < size - 1; i++)
      {
        MPI_Send(arrays[i % m], n, MPI_INT, (i + 1), i % m, MPI_COMM_WORLD);
      }
    }

    // TO-DO: arrays > ranks??

    // write results as processes send them
    for (i = 0; i < m; i++)
    {
      MPI_Recv(&temp, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
#if DEBUG
      printf("Rank %d found Max: %d\n", status.MPI_SOURCE, temp);
#endif
      int index = status.MPI_SOURCE - 1; // compensate for rank 0 being unused
      max[index] = temp;
    }
  }
  else if (rank <= m) // avoid calls to Recv/Send on unused processes
  {
    /* Allocate local array */
    array = malloc(sizeof(int) * n);
    if (array == NULL)
    {
      fprintf(stderr, "Unable to allocate %d elements for local array \n", m);
      return 1;
    }

    // waits until receiving work from rank 0, then individually works on its assigned array
    MPI_Recv(array, n, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
#if DEBUG
    printf("Rank %d received | arrays[%d][0]=%d\n", rank, status.MPI_TAG, array[0]);
#endif

    int _max = array[0];
    for (i = 0; i < n; i++)
    {
      if (array[i] > _max)
      {
        _max = array[i];
      }
    }
    MPI_Send(&_max, 1, MPI_INT, 0, rank, MPI_COMM_WORLD);
  }

  /* stop the measurement */
  t2 = MPI_Wtime();

  if (rank == 0)
  {
    printf("Computation time: %f s\n", t2 - t1);

    printf("(Seed: %d, Size: %d, # Arrays: %d) Time = %g s, \nMax values =\n", s, n, m, t2 - t1);
    for (i = 0; i < m; i++)
    {
      printf("%d\n", max[i]);
    }
  }

  MPI_Finalize();
  return 0;
}
