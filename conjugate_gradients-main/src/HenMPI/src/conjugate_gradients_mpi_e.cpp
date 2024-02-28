#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <omp.h>

#define first_touch_matrix
#define first_touch_rhs

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
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&num_rows, sizeof(size_t), 1, file);
    fread(&num_cols, sizeof(size_t), 1, file);
    
    if (rank==0 && num_cols != 1)
    {
        fprintf(stderr, "Right hand side has to have just a single column\n");
        return false;
    }

    rhs = new double[num_rows * num_cols];
    
    //first touch
#define first_touch_rhs
    #pragma omp parallel for
    for (int i = 0; i < num_rows; i++)
    {
        rhs[i] = 0;
    }
#endif // first_touch_rhs

    fread(rhs, sizeof(double), num_rows * num_cols, file);

    *rhs_out = rhs;
    *num_rows_out = num_rows;

    fclose(file);

    return true;
}

bool read_matrix_from_file(const char *filename, double **matrix_out, size_t *num_rows_out, size_t *num_cols_out)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double *matrix;
    size_t num_rows;
    size_t num_cols;
    MPI_File fhandle;
    unsigned int local_num_rows, offset;

    //Open the matrix file
    if(MPI_File_open(MPI_COMM_WORLD, file_matrix, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandle) != MPI_SUCCESS) {
        printf("[MPI process %d] Failure in opening the file.\n", myRank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    printf("[MPI process %d] File opened successfully.\n", myRank);

    // Read the matrix size
    MPI_File_read(fhandle, &num_rows, 1 ,MPI_UNSIGNED_LONG , MPI_STATUS_IGNORE);
    MPI_File_read(fhandle, &num_cols, 1 ,MPI_UNSIGNED_LONG , MPI_STATUS_IGNORE);
    std::cout << "Rank " << myRank << ": row = " << num_rows << " col = " << num_cols << std::endl;

    // calcolate the offset for each rank
    local_num_rows = num_rows / nRanks;
    offset = local_num_rows * sizeof(double) * myRank * num_cols;
    
    //the last rank will have the remaining rows
    if(myRank == nRanks - 1){
        local_num_rows += num_rows % nRanks; //add the reminder
    }
    std::cout << "Rank " << myRank << ": local_num_rows = " << local_num_rows << " offset = " << offset << std::endl;
    
    // seek the file to the correct position for each rank
    MPI_File_seek(fhandle, offset, MPI_SEEK_CUR);

    // Allocate memory for the local matrix
    matrix = new double[local_num_rows * num_cols];

#define first_touch_matrix
    //first touch 
    #pragma omp parallel for
    for(int i=0; i< local_num_rows * num_cols; i++){
        matrix[i] = 0;
    }
#endif // first_touch_matrix

    // Read the local matrix after first-touch
    MPI_File_read(fhandle, matrix, local_num_rows * num_cols, MPI_DOUBLE, MPI_STATUS_IGNORE);

    *matrix_out = matrix;
    *num_rows_out = local_num_rows;
    *num_cols_out = num_cols;

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

double dot(const double *x, const double *y, size_t size, size_t offset)
{

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double result = 0.0;
    double local_res = 0.0;
    
    #pragma omp parallel for reduction(+ : local_res)
    for (size_t i = offset; i < (offset+size); i++)
    {
        local_res += x[i] * y[i];
    }
    // Reduce the local results
    MPI_Allreduce(&local_res, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return result;
}

void axpby(double alpha, const double *x, double beta, double *y, size_t size)
{
    // y = alpha * x + beta * y
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

void gemv(double alpha, const double *A, const double *x, double beta, double *y, size_t num_rows, size_t num_cols, size_t offset)
{
    // y = alpha * A * x + beta * y;
    #pragma omp parallel for
    for (size_t r = 0; r < num_rows; r++)
    {
        double y_val = 0.0;
        for (size_t c = 0; c < num_cols; c++)
        {
            y_val += alpha * A[r * num_cols + c] * x[offset+c];
        }
        y[offset+r] = beta * y[offset+r] + y_val;
    }
    //gather the results
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, y, num_rows, MPI_DOUBLE, MPI_COMM_WORLD);
}

void conjugate_gradients(const double *A, const double *b, double *x, size_t rows, size_t cols, int max_iters, double rel_error)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double alpha, beta, bb, rr, rr_new;
    double *r = new double[rows];
    double *p = new double[rows];
    double *Ap = new double[rows];
    int num_iters;

    // calculate the offset for dot_product and gemv
    size_t offset = rank * cols;
    if(rank==num_procs-1){
        offset = rows-cols;
    }

    for (size_t i = 0; i < rows; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    bb = dot(b, b, rows);
    rr = bb;
    for (num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        gemv(1.0, A, p, 0.0, Ap, rows, cols, offset);
        alpha = rr / dot(p, Ap, cols, offset); 
        axpby(alpha, p, 1.0, x, rows, rank);
        axpby(-alpha, Ap, 1.0, r, rows);
        rr_new = dot(r, r, cols, offset);
        beta = rr_new / rr;
        rr = rr_new;

        // Print intermediate information
        if (rank == 0)
        {
            printf("Iteration %d: Residual norm = %e\n", num_iters, std::sqrt(rr / bb));
        }

        if (std::sqrt(rr / bb) < rel_error)
        {
            if (rank == 0)
            {
                printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / bb));
            }
            break;
        }
        axpby(1.0, r, beta, p, rows);
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

    printf("MPI+OMP Implementation, #Processes %d\n", num_procs);

    if(rank==0){
        printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
        printf("All parameters are optional and have default values\n");
        printf("\n");
    }

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

    if (rank == 0)
    {
        printf("Command line arguments:\n");
        printf("  input_file_matrix: %s\n", input_file_matrix);
        printf("  input_file_rhs:    %s\n", input_file_rhs);
        printf("  output_file_sol:   %s\n", output_file_sol);
        printf("  max_iters:         %d\n", max_iters);
        printf("  rel_error:         %e\n", rel_error);
        printf("\n");
    }

    double *local_matrix;
    double *rhs;
    size_t rows, cols;

    if (rank == 0)
    {

        printf("Reading local_matrix from file ...\n");
        bool success_read_local_matrix = read_matrix_from_file(input_file_matrix, &local_matrix, &rows, &cols);
        if (!success_read_local_matrix)
        {
            fprintf(stderr, "Failed to read local matrix\n");
            return 1;
        }

        printf("Process %d -> Read local matrix Done\n", rank);
        printf("\n");

        if (rank == 0)
        {
            printf("Reading right hand side from file ...\n");
        }
        size_t rhs_rows;
        bool success_read_rhs = read_rhs_from_file(input_file_rhs, &rhs, &rhs_rows);
        if (!success_read_rhs)
        {
            fprintf(stderr, "Failed to read right hand side\n");
            return 2;
        }
        printf("Process %d -> Reading RHS Done\n", rank);
        printf("\n");

        if (rhs_rows != matrix_rows)
        {
            fprintf(stderr, "Size of right hand side does not match the matrix\n");
            return 4;
        }
    }

    if (rank == 0)
    {
        printf("Solving the system ...\n");
    }

    double * sol = new double[rows];
    conjugate_gradients(local_matrix, rhs, sol, rows, cols, max_iters, rel_error);
    printf("Done\n");
    printf("\n");

    if(rank==0){
        printf("Writing solution to file ...\n");
        bool success_write_sol = write_matrix_to_file(output_file_sol, sol, rows, 1);
        if(!success_write_sol)
        {
            fprintf(stderr, "Failed to save solution\n");
            return 6;
        }
        printf("Done\n");
        printf("\n");
    }
    
    delete[] local_matrix;
    delete[] rhs;
    delete[] sol;

    printf("Finished successfully\n");

    return 0;
}