// Your First C++ Program

#include <iostream>
#include <stdio.h>
#include <cmath>

int main(int argc, char *argv[])
{
    int n = atoi(argv[1]);
    int sqrt_n = (int)sqrt(n);

    std::cout << "Generating mesh with " << n << " points" << std::endl;

    FILE *fp = fopen("big_mesh_c.off", "w");
    
    fprintf(fp, "OFF\n");
    fprintf(fp, "%d %d %d\n", n, int(2*(sqrt_n-1)*(sqrt_n-1)),int(0));

    std::cout << "Writing points..." << std::endl;

    for (int i = 0; i < sqrt_n; i++)
    {
        for (int j = 0; j < sqrt_n; j++)
        {
            fprintf(fp, "%f %f %f\n", (float)i, (float)j, 0.0f);
        }
    }

    std::cout << "Writing faces..." << std::endl;

    for (int i = 0; i < n-sqrt_n; i++)
    {
        if (i % sqrt_n != sqrt_n-1){
            fprintf(fp, "3 %d %d %d\n", i, i+1, i+sqrt_n+1);
            fprintf(fp, "3 %d %d %d\n", i, i+sqrt_n+1, i+sqrt_n);
        }
    }

    std::cout << "Done!" << std::endl;

    return 0; 
}
