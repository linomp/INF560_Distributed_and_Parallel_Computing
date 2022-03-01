#include <omp.h>
#include <stdio.h>

void main()
{
    printf("# Threads: % d\n", omp_get_num_threads());

#pragma omp parallel
    {
        printf(
            "Hello from thread %d/%d\n",
            omp_get_thread_num(),
            omp_get_num_threads());
    }
}
