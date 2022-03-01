#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <limits.h>

/* MPI function signatures */
#include <mpi.h>

int main(int argc, char **argv)
{
    int N, me;
    char hostname[HOST_NAME_MAX + 1];

    /* Initialization of MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &N);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    gethostname(hostname, HOST_NAME_MAX + 1);

    printf("Hello World from task %d out of %d on %s\n", me, N, hostname);
    MPI_Finalize();
    return 0;
}