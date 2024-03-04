#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cuda.h>

#include "LAM.hpp"

#define PRINT_RANK0(...) if(myRank==0) printf(__VA_ARGS__)
#define PRINT_ERR_RANK0(...) if(myRank==0) fprintf(stderr, __VA_ARGS__)

using namespace LAM;

int main(int argc, char ** argv)
{
    printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    printf("All parameters are optional and have default values\n");
    printf("\n");

    const char * input_file_matrix = "io/matrix.bin";
    const char * input_file_rhs = "io/rhs.bin";
    const char * output_file_sol = "io/sol.bin";
    int max_iters = 1000;
    double rel_error = 1e-9;

    if(argc > 1) input_file_matrix = argv[1];
    if(argc > 2) input_file_rhs = argv[2];
    if(argc > 3) output_file_sol = argv[3];
    if(argc > 4) max_iters = atoi(argv[4]);
    if(argc > 5) rel_error = atof(argv[5]);

    printf("Command line arguments:\n");
    printf("  input_file_matrix: %s\n", input_file_matrix);
    printf("  input_file_rhs:    %s\n", input_file_rhs);
    printf("  output_file_sol:   %s\n", output_file_sol);
    printf("  max_iters:         %d\n", max_iters);
    printf("  rel_error:         %e\n", rel_error);
    printf("\n");


    using namespace std::chrono;

    ConjugateGradient_MultiGPUS_CUDA<double> CG_P;

    {
        printf("Reading matrix from file ...\n");
        bool success_read_matrix = CG_P.load_matrix_from_file(input_file_matrix);
        if(!success_read_matrix)
        {
            printf("Failed to read matrix\n");
            return 1;
        }
        printf("Done\n");
        printf("\n");

        printf("Reading right hand side from file ...\n");
        bool success_read_rhs = CG_P.load_rhs_from_file(input_file_rhs);
        if(!success_read_rhs)
        {
            printf("Failed to read right hand side\n");
            return 2;
        }
        printf("Done\n");
        printf("\n");
    }

    printf("Solving the system ...\n");

    auto start = high_resolution_clock::now();
    CG_P.solve(max_iters, rel_error);
    auto end =  high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);

    printf("Time elapsed using parallel implementation:%f s\n", duration.count()/1000.0);
    // sequential execution
/*
    ConjugateGradient<double> CG_baseline(std::make_unique<MelBLAS_baseline<double>>());

    start = high_resolution_clock::now();
    CG_baseline.solve(A, b, x, size, max_iters, rel_error);
    end =  high_resolution_clock::now();

    auto duration_baseline = duration_cast<milliseconds>(end - start);

    std::cout << "Time elapsed using baseline implementation:" << duration_baseline.count()/1000.0 << " s"<< std::endl;

    std::cout << "Total speedup:" << duration_baseline.count()/duration.count() << std::endl;
*/
    printf("Done\n");
    printf("\n");

    printf("Writing solution to file ...\n");
    bool success_write_sol = CG_P.save_result_to_file(output_file_sol);
    if(!success_write_sol)
    {
        printf("Failed to save solution\n");
        return 6;
    }
    printf("Done\n");
    printf("\n");

    printf("Finished successfully\n");

    return 0;
}