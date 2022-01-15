#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
  int rank, size;
  int s;
  int n;
  int *tab;
  int i;
  int max;
  int temp;
  double t1, t2;

  int *partials; // TODO: better approach, no extra array...
  int t;
  MPI_Status status;

  /* MPI Initialization */
  MPI_Init(&argc, &argv);

  /* Get the rank of the current task and the number
   * of MPI processe
   */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Check the input arguments */
  if (argc < 3)
  {
    printf("Usage: %s S N\n", argv[0]);
    printf("\tS: seed for pseudo-random generator\n");
    printf("\tN: size of the array\n");
    exit(1);
  }

  s = atoi(argv[1]);
  n = atoi(argv[2]);
  srand48(s);

  partials = calloc(size, sizeof(int));

  /* Allocate the array */
  tab = malloc(sizeof(int) * n);
  if (tab == NULL)
  {
    fprintf(stderr, "Unable to allocate %d elements\n", n);
    return 1;
  }

  /* Initialize the array */
  for (i = 0; i < n; i++)
  {
    tab[i] = lrand48() % n;
  }

  /* start the measurement */
  t1 = MPI_Wtime();

  /* search for the max value */
  max = tab[(rank * n / size)];
  for (i = (rank * n / size); i < ((rank * n / size) + (n / size)); i++)
  {
    if (tab[i] > max)
    {
      max = tab[i];
    }
  }

  if (rank == 0)
  {
    partials[0] = max;

    for (t = 1; t < size; t++)
    {
      MPI_Recv(&temp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      partials[t] = temp;
#if DEBUG
      for (i = 0; i < size; i++)
      {
        printf("%d  ", partials[i]);
      }
      printf("Message from rank %d: %d\n", status.MPI_SOURCE, temp);
#endif
    }

    /* search for the max among partial results */
    max = partials[0];
    for (i = 0; i < size; i++)
    {
      if (partials[i] > max)
      {
        max = partials[i];
      }
    }
  }
  else
  {
    MPI_Send(&max, 1, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
  }

  /* stop the measurement */
  t2 = MPI_Wtime();

  if (rank == 0)
  {
    printf("(Seed %d, Size %d) Max value = %d, Time = %g s\n", s, n, max, t2 - t1);

    printf("Computation time: %f s\n", t2 - t1);
  }

  MPI_Finalize();
  return 0;
}
