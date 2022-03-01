#include <stdio.h>

/* MPI function signatures */
#include <mpi.h>

int main(int argc, char **argv)
{
    int N, me, t;
    int buff;

    MPI_Status status;

    /* Initialization of MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &N);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);

    if (me == 0)
    {
        // wait for N-1 messages
        for (t = 1; t < N; t++)
        {
            MPI_Recv(&buff, 1, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
        printf("Hello World with %d ready tasks\n", N);
    }
    else
    {
        // send message confirming task is ready
        MPI_Send(&buff, 1, MPI_DOUBLE, 0, me, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}