#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* Function computing the final string to print */
void compute_string(char *res, char *a, char *b, char *c, int length)
{
    int i;

    for (i = 0; i < length; i++)
    {
        res[i] = a[i] + b[i] + c[i];
    }
}

int main()
{

    char *res;

    char a[30] = {40, 70, 70, 70, 80, 0, 50, 80, 80, 70, 70, 0, 40, 80, 79, 70, 0, 40, 50, 50, 0, 70, 80, 0, 30, 50, 30, 30, 0, 0};
    char b[30] = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 0, 0};
    char c[30] = {22, 21, 28, 28, 21, 22, 27, 21, 24, 28, 20, 22, 20, 24, 22,
                  29, 22, 21, 20, 25, 22, 25, 20, 22, 27, 25, 28, 25, 0, 0};

    res = (char *)malloc(30 * sizeof(char));

    /* This function call should be programmed in CUDA */
    /* -> need to allocate and transfer data to/from the device */
    compute_string(res, a, b, c, 30);

    printf("%s\n", res);

    return 0;
}
