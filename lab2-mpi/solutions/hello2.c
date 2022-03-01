/**
 * INF560 - TD2
 *
 * Part 1-b
 */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char**argv) {
  int rank, size;
  int ready = 0;

  MPI_Init(&argc, &argv);

  /* Grab the rank of the current MPI task */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Grab the total number of MPI tasks */
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  /* Each rank (but 0) send a message to tell the rank 0
   * that it is ready.
   * The content of this message will not be used
   */
  if(rank>0) {

    MPI_Send(&ready, 1, MPI_INTEGER, 0, 0, MPI_COMM_WORLD);

  } else {
    MPI_Status status;
    int i;

    /* Rank 0 will gather all messages (size-1) */
    for(i=1; i<size; i++) {
      MPI_Recv(&ready, 1, MPI_INTEGER, i, 0, MPI_COMM_WORLD, &status);
    }

    fprintf(stderr, "Hello world with %d ready tasks\n", size);
  }

  MPI_Finalize();
  return 0;
}
