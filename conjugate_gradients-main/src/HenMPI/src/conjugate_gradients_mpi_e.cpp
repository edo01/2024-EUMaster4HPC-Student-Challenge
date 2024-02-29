#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <omp.h>
#include <iostream>

#define PRINT_RANK0(...) if(rank==1) printf(__VA_ARGS__)
#define PRINT_ERR_RANK0(...) if(rank==0) fprintf(stderr, __VA_ARGS__)

//#define first_touch_matrix
//#define first_touch_rhs

bool read_rhs_from_file(const char *filename, double **rhs_out, size_t *num_rows_out)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double *rhs;
    size_t num_rows;
    size_t num_cols;

    FILE *file = fopen(filename, "rb");
    if (file == nullptr)
    {
        PRINT_ERR_RANK0("Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    
    if (num_cols != 1)
    {
        PRINT_ERR_RANK0("Right hand side has to have just a single column\n");
        return false;
    }

    rhs = new double[num_rows];
    
    //first touch
#ifdef first_touch_rhs
    #pragma omp parallel for
    for (int i = 0; i < num_rows; i++)
    {
        rhs[i] = 0;
    }
#endif // first_touch_rhs

    fread(rhs, sizeof(double), num_rows, file);

    *rhs_out = rhs;
    *num_rows_out = num_rows;

    fclose(file);

    return true;
}

bool read_matrix_from_file(const char *filename, double **matrix_out, size_t *num_local_rows_out, 
                           size_t *num_cols_out, int **sendcounts_out, int **displs_out)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double *matrix;
    size_t num_rows;
    size_t num_cols;
    size_t num_local_rows;
    size_t offset;
    int* sendcounts;
    int* displs;
    MPI_File fhandle;

    /*
    ----------------------------------------------
                Open the file
    ----------------------------------------------
    */
    if(MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandle) != MPI_SUCCESS) {
        PRINT_ERR_RANK0("Failed to open file\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return false;
    }
    printf("[MPI process %d] File opened successfully.\n", rank);
    
    // Read the matrix size
    MPI_File_read(fhandle, &num_rows, 1 ,MPI_UNSIGNED_LONG , MPI_STATUS_IGNORE);
    MPI_File_read(fhandle, &num_cols, 1 ,MPI_UNSIGNED_LONG , MPI_STATUS_IGNORE);

    // calcolate the offset for each rank
    num_local_rows = num_rows / num_procs;
    offset = num_local_rows * sizeof(double) * rank * num_cols;
    
    //the last rank will have the remaining rows
    if(rank == num_procs - 1){
        //add the reminder to the last rank
        num_local_rows += num_rows % num_procs; 
    }

    //calculate the displacement and the sendcounts for the scatterv
    sendcounts = new int[num_procs]; 
    displs = new int[num_procs];

    for(int i=0; i<num_procs; i++){
        sendcounts[i] = (i==num_procs-1) ? num_rows/num_procs + num_rows%num_procs : num_rows/num_procs;
        displs[i] = i * num_rows/num_procs;
        PRINT_RANK0("sendcounts[%d] = %d, displs[%d] = %d\n", i, sendcounts[i], i, displs[i]);
    }

    // seek the file to the correct position for each rank
    MPI_File_seek(fhandle, offset, MPI_SEEK_CUR);

    // Allocate memory for the local matrix
    matrix = new double[num_local_rows * num_cols];

#ifdef first_touch_matrix
    //first touch 
    #pragma omp parallel for
    for(int i=0; i< num_local_rows * num_cols; i++){
        matrix[i] = 0;
    }
#endif // first_touch_matrix

    // Read the local matrix after first-touch
    MPI_File_read(fhandle, matrix, num_local_rows * num_cols, MPI_DOUBLE, MPI_STATUS_IGNORE);

    *matrix_out = matrix;
    *num_local_rows_out = num_local_rows;
    *num_cols_out = num_cols;
    *sendcounts_out = sendcounts;
    *displs_out = displs;

    MPI_File_close(&fhandle);
    
    return true;
}

bool write_matrix_to_file(const char *filename, const double *matrix, size_t num_rows, size_t num_cols)
{
    FILE *file = fopen(filename, "wb");
    if (file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fwrite(&num_rows, sizeof(size_t), 1, file);
    fwrite(&num_cols, sizeof(size_t), 1, file);
    fwrite(matrix, sizeof(double), num_rows * num_cols, file);

    fclose(file);

    return true;
}

void print_matrix(const double *matrix, size_t num_rows, size_t num_cols, FILE *file = stdout)
{
    fprintf(file, "%zu %zu\n", num_rows, num_cols);
    for (size_t r = 0; r < num_rows; r++)
    {
        for (size_t c = 0; c < num_cols; c++)
        {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}

double dot(const double *x, const double *y, size_t local_row, size_t offset)
{

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double result = 0.0;
    double local_res = 0.0;

    //#pragma omp parallel for reduction(+ : local_res)
    for (size_t i = 0; i < local_row; i++)
    {
        local_res += x[offset+i] * y[offset+i];
    }
    
    // Reduce the local results
    MPI_Allreduce(&local_res, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    return result;
}

void axpby(double alpha, const double *x, double beta, double *y, size_t size)
{
    // y = alpha * x + beta * y

    //#pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

void gemv(double alpha, const double *A, const double *x, double beta, double *y,
            size_t num_local_rows, size_t num_cols, size_t offset, const int *sendcounts, const int *displs)
{
    // y = alpha * A * x + beta * y;

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double *y_temp = new double[num_cols];
    double y_val;

    //#pragma omp parallel for
    for (size_t r = 0; r < num_local_rows; r++)
    {
        //row-col multiplication
        y_val = 0.0;
        for (size_t c = 0; c < num_cols; c++)
        {
            y_val += alpha * A[r * num_cols + c] * x[c];
        }
        //saving the value in the correct position
        y_temp[offset+r] = beta * y[offset+r] + y_val;
    }

    MPI_Allgatherv(y_temp+offset, num_local_rows, MPI_DOUBLE, y, sendcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    

    

    delete[] y_temp;
}

void conjugate_gradients(const double *A, const double *b, double *x, size_t local_row, size_t cols, 
                         const int *sendcounts, const int *displs , int max_iters, double rel_error)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int num_iters;
    double alpha, beta, bb, rr, rr_new;
    double *r = new double[cols];
    double *p = new double[cols];
    double *Ap = new double[cols];

    //first touch?

    // calculate the offset for dot_product and gemv
    size_t offset = local_row * rank; //THIS IS WRONG

    //print the offset
    printf("rank %d -> offset = %zu\n", rank, offset);

    //first touch
    for (size_t i = 0; i < cols; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    bb = dot(b, b, local_row, offset);
    //print bb
    printf("bb = %e\n", bb);
    
    rr = bb;
    for (num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        gemv(1.0, A, p, 0.0, Ap, local_row, cols, offset, sendcounts, displs);
        alpha = rr / dot(p, Ap, local_row, offset); 
        axpby(alpha, p, 1.0, x, cols);
        axpby(-alpha, Ap, 1.0, r, cols);
        rr_new = dot(r, r, local_row, offset);
        beta = rr_new / rr;
        rr = rr_new;

        //print the relative error at each iteration
        if(rank==0){
            //printf("Iteration %d, relative error is %e\n", num_iters, std::sqrt(rr / bb));
        } 

        if (std::sqrt(rr / bb) < rel_error)
        {
            if (rank == 0)
            {
                printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
            }
            break;
        }
        axpby(1.0, r, beta, p, cols);
    }

    delete[] r;
    delete[] p;
    delete[] Ap;

    if (num_iters > max_iters)
    {
        if (rank == 0)
        {
            printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / bb));
        }
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    /*
    ----------------------------------------------
                Initialize MPI environment
    ----------------------------------------------
    */

    PRINT_RANK0("MPI+OMP Implementation, #Processes %d\n", num_procs);
    PRINT_RANK0("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    PRINT_RANK0("All parameters are optional and have default values\n");
    PRINT_RANK0("\n");
    

    const char *input_file_matrix = "io/matrix.bin";
    const char *input_file_rhs = "io/rhs.bin";
    const char *output_file_sol = "io/sol.bin";
    int max_iters = 1000;
    double rel_error = 1e-9;

    if (argc > 1)
        input_file_matrix = argv[1];
    if (argc > 2)
        input_file_rhs = argv[2];
    if (argc > 3)
        output_file_sol = argv[3];
    if (argc > 4)
        max_iters = atoi(argv[4]);
    if (argc > 5)
        rel_error = atof(argv[5]);

    PRINT_RANK0("Command line arguments:\n");
    PRINT_RANK0("  input_file_matrix: %s\n", input_file_matrix);
    PRINT_RANK0("  input_file_rhs:    %s\n", input_file_rhs);
    PRINT_RANK0("  output_file_sol:   %s\n", output_file_sol);
    PRINT_RANK0("  max_iters:         %d\n", max_iters);
    PRINT_RANK0("  rel_error:         %e\n", rel_error);
    PRINT_RANK0("\n");
    PRINT_RANK0("Reading local_matrix from file ...\n");
    

    /*
    ----------------------------------------------
    Read the local matrix and the right hand side
    ----------------------------------------------
    */
    double *local_matrix;
    double *rhs;
    size_t local_rows, cols, rhs_rows;
    int *sendcounts, *displs;

    bool success_read_local_matrix = 
        read_matrix_from_file(input_file_matrix, &local_matrix, &local_rows, &cols, &sendcounts, &displs);
    
    if (!success_read_local_matrix)
    {
        PRINT_ERR_RANK0("Failed to read local matrix\n");
        return 1;
    }

    printf("Process %d -> Read local matrix Done\n", rank);
    printf("\n");

    PRINT_RANK0("Reading right hand side from file ...\n");

    bool success_read_rhs = read_rhs_from_file(input_file_rhs, &rhs, &rhs_rows);
    if (!success_read_rhs)
    {
        PRINT_ERR_RANK0("Failed to read right hand side\n");
        return 2;
    }
    printf("Process %d -> Reading RHS Done\n", rank);
    printf("\n");

    if (rhs_rows != cols)
    {
        PRINT_ERR_RANK0( "Size of right hand side does not match the matrix\n");
        return 4;
    }

    /*
    ----------------------------------------------
                Solve the system
    ----------------------------------------------
    */
    PRINT_RANK0("Solving the system ...\n");
    
    double * sol = new double[cols];
    conjugate_gradients(local_matrix, rhs, sol, local_rows, cols, sendcounts, displs, max_iters, rel_error);
    PRINT_RANK0("Done\n");
    PRINT_RANK0("\n");

    /*
    ----------------------------------------------
                Save the solution to file
    ----------------------------------------------
    */
    if(rank==0){
        printf("Writing solution to file ...\n");
        bool success_write_sol = write_matrix_to_file(output_file_sol, sol, cols, 1);
        if(!success_write_sol)
        {
            fprintf(stderr, "Failed to save solution\n");
            return 6;
        }
        printf("Done\n");
        printf("\n");
    }

    /*
    ----------------------------------------------
                        Clean up
    ----------------------------------------------
    */
    
    delete[] local_matrix;
    delete[] rhs;
    delete[] sol;

    PRINT_RANK0("Finished successfully\n");

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}