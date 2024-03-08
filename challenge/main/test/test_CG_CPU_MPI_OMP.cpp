#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>


#include "LAM.hpp"

#define PRINT_RANK0_V(...) if(rank==0 && verbose) printf(__VA_ARGS__)

using namespace LAM;


int rank, size;
const char * input_file_matrix = "io/matrix.bin";
const char * input_file_rhs = "io/rhs.bin";
const char * output_file_sol = "io/sol.bin";
int max_iters = 10000;
double rel_error = 1e-9;
size_t rows=0, cols=0;
bool verbose = false;


int load_mode(){

    PRINT_RANK0_V("Command line arguments:\n");
    PRINT_RANK0_V("  input_file_matrix: %s\n", input_file_matrix);
    PRINT_RANK0_V("  input_file_rhs:    %s\n", input_file_rhs);
    PRINT_RANK0_V("  output_file_sol:   %s\n", output_file_sol);
    PRINT_RANK0_V("  max_iters:         %d\n", max_iters);
    PRINT_RANK0_V("  rel_error:         %e\n", rel_error);
    //print number of processes
    PRINT_RANK0_V("  Number of processes: %d\n", size);
    //print number of threads
    PRINT_RANK0_V("  Number of threads: %d\n", omp_get_max_threads());

    PRINT_RANK0_V("\n");


    using namespace std::chrono;

    ConjugateGradient_CPU_MPI_OMP<double> CG_P;

    {
        PRINT_RANK0_V("Reading matrix from file ...\n");
        auto start = high_resolution_clock::now();
        bool success_read_matrix = CG_P.load_matrix_from_file(input_file_matrix);
        auto end =  high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        if(!success_read_matrix)
        {
            PRINT_ERR_RANK0("Failed to read matrix\n");
            return 1;
        }

        /*PRINT MPI proc, OMP threads, file reading*/
        //PRINT_RANK0_V("%d,%d,file:%f,", size, omp_get_max_threads(),duration.count());
        if(rank==0 && !verbose){
            std::cout<<size<<","<<omp_get_max_threads()<<","<<duration.count()/1000.0<<",";
        }

        PRINT_RANK0_V("Done\n");
        PRINT_RANK0_V("\n");

        PRINT_RANK0_V("Reading right hand side from file ...\n");
        bool success_read_rhs = CG_P.load_rhs_from_file(input_file_rhs);
        if(!success_read_rhs)
        {
            PRINT_ERR_RANK0("Failed to read right hand side\n");
            return 2;
        }
        PRINT_RANK0_V("Done\n");
        PRINT_RANK0_V("\n");
    }

    PRINT_RANK0_V("Solving the system ...\n");

    auto start_cg = high_resolution_clock::now();
    CG_P.solve(max_iters, rel_error);
    auto end_cg =  high_resolution_clock::now();

    auto duration_cg = duration_cast<milliseconds>(end_cg - start_cg);
    
    // Print CG duration
    if(rank==0 && !verbose){
        std::cout<<duration_cg.count()/1000.0;
    }

    //PRINT_RANK0_V("Time elapsed using parallel implementation:%f s\n", duration_cg.count()/1000.0);
    PRINT_RANK0_V("Done\n");
    PRINT_RANK0_V("\n");

    PRINT_RANK0_V("Writing solution to file ...\n");
    bool success_write_sol = CG_P.save_result_to_file(output_file_sol);
    if(!success_write_sol)
    {
        PRINT_ERR_RANK0("Failed to save solution\n");
        return 6;
    }

    PRINT_RANK0_V("Done\n");
    PRINT_RANK0_V("\n");

    PRINT_RANK0_V("Finished successfully\n");

    return 0;
}

int gen_mode(){

    PRINT_RANK0_V("Command line arguments:\n");
    PRINT_RANK0_V("  rows:    %lu\n", rows);
    PRINT_RANK0_V("  cols:    %lu\n", cols);
    PRINT_RANK0_V("  size of the problem: %f GB\n", rows*cols*sizeof(double)/1024.0/1024.0/1024.0);
    PRINT_RANK0_V("  output_file_sol:   %s\n", output_file_sol);
    PRINT_RANK0_V("  max_iters:         %d\n", max_iters);
    PRINT_RANK0_V("  rel_error:         %e\n", rel_error);
    //print number of processes
    PRINT_RANK0_V("  Number of processes: %d\n", size);
    //print number of threads
    PRINT_RANK0_V("  Number of threads: %d\n", omp_get_max_threads());

    PRINT_RANK0_V("\n");

    using namespace std::chrono;

    ConjugateGradient_CPU_MPI_OMP<double> CG_P;

    {
        auto start = high_resolution_clock::now();
        bool success_read_matrix = CG_P.generate_matrix(rows, cols);
        auto end =  high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);
        if(!success_read_matrix)
        {
            PRINT_ERR_RANK0("Failed to read matrix\n");
            return 1;
        }
        
        PRINT_RANK0_V("Time elapsed for generating the matrix:%f s\n", duration.count()/1000.0);
        PRINT_RANK0_V("Done\n");
        PRINT_RANK0_V("\n");

        /*PRINT MPI proc, OMP threads, file reading*/
        if(rank==0 && !verbose){
            std::cout<<size<<","<<omp_get_max_threads()<<","<<duration.count()/1000.0<<",";
            fflush(stdout);
        }
        fflush(stdout);

        PRINT_RANK0_V("Reading right hand side from file ...\n");
        bool success_read_rhs = CG_P.generate_rhs();
        if(!success_read_rhs)
        {
            PRINT_ERR_RANK0("Failed to read right hand side\n");
            return 2;
        }
        PRINT_RANK0_V("Done\n");
        PRINT_RANK0_V("\n");
    }

    PRINT_RANK0_V("Solving the system ...\n");

    auto start_cg = high_resolution_clock::now();
    CG_P.solve(max_iters, rel_error);
    auto end_cg =  high_resolution_clock::now();

    auto duration = duration_cast<milliseconds>(end_cg - start_cg);

    PRINT_RANK0_V("cg_tot:%f,\n", duration.count());
    if(rank==0 && !verbose){
        std::cout<<duration.count()/1000;
    }
    fflush(stdout);
 
    //PRINT_RANK0_V("Time elapsed using parallel implementation:%f s\n", duration.count()/1000.0);
    PRINT_RANK0_V("Done\n");
    PRINT_RANK0_V("\n");

    //PRINT_RANK0_V("Writing solution to file ...\n");
    bool success_write_sol = CG_P.save_result_to_file(output_file_sol);
    if(!success_write_sol)
    {
        PRINT_ERR_RANK0("Failed to save solution\n");
        return 6;
    }
    PRINT_RANK0_V("Done\n");
    PRINT_RANK0_V("\n");

    PRINT_RANK0_V("Finished successfully\n");

    return 0;

}

/*
* Output: num_rows, mpi_proc, omp_threads, file_reading_time, avg_gemv, avg_one_iter, num_iter, err, cg_time
*/

int main(int argc, char ** argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    bool mode_generate = false;
    bool mode_load = false;
    
    int opt;

    while((opt = getopt(argc, argv, "hvA:b:o:i:e:s:")) != -1) {
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
            case 'v':
                verbose = true;
                break;
            case 'h':
                PRINT_RANK0("Usage: ./test_CG_CPU_MPI_OMP_gen.out [ (-A -b | -s) -o -e -i -h -v]\n");
                PRINT_RANK0("Options:\n");
                PRINT_RANK0("  -A <file>       Read matrix from file\n");
                PRINT_RANK0("  -b <file>       Read right hand side from file\n");
                PRINT_RANK0("  -o <file>       Write solution to file\n");
                PRINT_RANK0("  -i <int>        Maximum number of iterations\n");
                PRINT_RANK0("  -e <float>      Relative error\n");
                PRINT_RANK0("  -s <int>        Generate matrix of size n x n\n");
                PRINT_RANK0("  -v              Verbose mode\n");
                PRINT_RANK0("  -h              Show this help message\n");
                return 0;
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
    std::cout<<std::endl;

    MPI_Finalize();

    return res;
} 