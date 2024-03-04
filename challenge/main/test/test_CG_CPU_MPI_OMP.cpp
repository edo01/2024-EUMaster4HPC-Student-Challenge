#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <unistd.h>


#include "LAM.hpp"

#define PRINT_RANK0(...) if(rank==0) printf(__VA_ARGS__)
#define PRINT_ERR_RANK0(...) if(rank==0) fprintf(stderr, __VA_ARGS__)

using namespace LAM;


int rank, size;
const char * input_file_matrix = "io/matrix.bin";
const char * input_file_rhs = "io/rhs.bin";
const char * output_file_sol = "io/sol.bin";
int max_iters = 100000;
double rel_error = 1e-8;
size_t rows=0, cols=0;


int load_mode(){

    PRINT_RANK0("Command line arguments:\n");
    PRINT_RANK0("  input_file_matrix: %s\n", input_file_matrix);
    PRINT_RANK0("  input_file_rhs:    %s\n", input_file_rhs);
    PRINT_RANK0("  output_file_sol:   %s\n", output_file_sol);
    PRINT_RANK0("  max_iters:         %d\n", max_iters);
    PRINT_RANK0("  rel_error:         %e\n", rel_error);
    PRINT_RANK0("\n");


    using namespace std::chrono;

    ConjugateGradient_CPU_MPI_OMP<double> CG_P;

    {
        PRINT_RANK0("Reading matrix from file ...\n");
        bool success_read_matrix = CG_P.load_matrix_from_file(input_file_matrix);
        if(!success_read_matrix)
        {
            PRINT_ERR_RANK0("Failed to read matrix\n");
            return 1;
        }
        PRINT_RANK0("Done\n");
        PRINT_RANK0("\n");

        PRINT_RANK0("Reading right hand side from file ...\n");
        bool success_read_rhs = CG_P.load_rhs_from_file(input_file_rhs);
        if(!success_read_rhs)
        {
            PRINT_ERR_RANK0("Failed to read right hand side\n");
            return 2;
        }
        PRINT_RANK0("Done\n");
        PRINT_RANK0("\n");
    }

    PRINT_RANK0("Solving the system ...\n");

    auto start = high_resolution_clock::now();
    CG_P.solve(max_iters, rel_error);
    auto end =  high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);
 
    PRINT_RANK0("Time elapsed using parallel implementation:%f s\n", duration.count()/1000.0);
    PRINT_RANK0("Done\n");
    PRINT_RANK0("\n");

    PRINT_RANK0("Writing solution to file ...\n");
    bool success_write_sol = CG_P.save_result_to_file(output_file_sol);
    if(!success_write_sol)
    {
        PRINT_ERR_RANK0("Failed to save solution\n");
        return 6;
    }
    PRINT_RANK0("Done\n");
    PRINT_RANK0("\n");

    PRINT_RANK0("Finished successfully\n");

    return 0;
}

int gen_mode(){

    PRINT_RANK0("Command line arguments:\n");
    PRINT_RANK0("  rows:    %lu\n", rows);
    PRINT_RANK0("  cols:    %lu\n", cols);
    PRINT_RANK0("  output_file_sol:   %s\n", output_file_sol);
    PRINT_RANK0("  max_iters:         %d\n", max_iters);
    PRINT_RANK0("  rel_error:         %e\n", rel_error);
    PRINT_RANK0("\n");


    using namespace std::chrono;

    ConjugateGradient_CPU_MPI_OMP<double> CG_P;

    {
        PRINT_RANK0("Reading matrix from file ...\n");
        bool success_read_matrix = CG_P.generate_matrix(rows, cols);
        if(!success_read_matrix)
        {
            PRINT_ERR_RANK0("Failed to read matrix\n");
            return 1;
        }
        PRINT_RANK0("Done\n");
        PRINT_RANK0("\n");

        PRINT_RANK0("Reading right hand side from file ...\n");
        bool success_read_rhs = CG_P.generate_rhs();
        if(!success_read_rhs)
        {
            PRINT_ERR_RANK0("Failed to read right hand side\n");
            return 2;
        }
        PRINT_RANK0("Done\n");
        PRINT_RANK0("\n");
    }

    PRINT_RANK0("Solving the system ...\n");

    auto start = high_resolution_clock::now();
    CG_P.solve(max_iters, rel_error);
    auto end =  high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end - start);
 
    PRINT_RANK0("Time elapsed using parallel implementation:%f s\n", duration.count()/1000.0);
    PRINT_RANK0("Done\n");
    PRINT_RANK0("\n");

    PRINT_RANK0("Writing solution to file ...\n");
    bool success_write_sol = CG_P.save_result_to_file(output_file_sol);
    if(!success_write_sol)
    {
        PRINT_ERR_RANK0("Failed to save solution\n");
        return 6;
    }
    PRINT_RANK0("Done\n");
    PRINT_RANK0("\n");

    PRINT_RANK0("Finished successfully\n");

    return 0;

}

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    PRINT_RANK0("Usage: ./test_CG_CPU_MPI_OMP_gen.out [ (-A -b | -s) -o -e -i]\n");
    PRINT_RANK0("\n");

    bool mode_generate = false;
    bool mode_load = false;
    
    int opt;
    while((opt = getopt(argc, argv, "A:b:o:i:e:s:")) != -1) {
        switch(opt) {
            case 'A':
                if(mode_generate) {
                    fprintf(stderr, "Option -s cannot be used with -A.\n");
                    return 1;
                }
                mode_load = true;
                input_file_matrix = optarg;
                break;
            case 'b':
                if(mode_generate) {
                    fprintf(stderr, "Option -s cannot be used with -b.\n");
                    return 1;
                }
                mode_load = true;
                input_file_rhs = optarg;
                break;
            case 'o':
                output_file_sol = optarg;
                break;
            case 'i':
                max_iters = atoi(optarg);
                break;
            case 'e':
                rel_error = atof(optarg);
                break;
            case 's':
                if(mode_load) {
                    fprintf(stderr, "Option -A and -b cannot be used with -s.\n");
                    return 1;
                }
                mode_generate = true;
                rows = atoi(optarg);
                cols = rows;
                break;
            case '?':
                if (optopt == 'A' || optopt == 'b' || optopt == 'o' || optopt == 'i' || optopt == 'e' || optopt == 's') {
                    fprintf(stderr, "Option -%c requires an argument.\n", optopt);
                } else if (isprint(optopt)) {
                    fprintf(stderr, "Unknown option `-%c'.\n", optopt);
                } else {
                    fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
                }
                return 1;
            default:
                abort();
        }
    }

    int res;
    if(mode_generate){
        res=gen_mode();
    }else if(mode_load){
        res=load_mode();
    }

    MPI_Finalize();

    return res;
} 