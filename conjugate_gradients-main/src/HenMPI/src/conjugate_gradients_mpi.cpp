#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mpi.h>
#include <omp.h>

bool read_matrix_from_file(const char *filename, double **matrix_out, size_t *num_rows_out, size_t *num_cols_out)
{
    double *matrix;
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
    matrix = new double[num_rows * num_cols];
    fread(matrix, sizeof(double), num_rows * num_cols, file);

    *matrix_out = matrix;
    *num_rows_out = num_rows;
    *num_cols_out = num_cols;

    fclose(file);

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
#pragma omp parallel for
        for (size_t c = 0; c < num_cols; c++)
        {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}

double dot(const double *x, const double *y, size_t size, MPI_Comm comm)
{
    double result = 0.0;
    double local = 0.0;
#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    // MPI_Allreduce(&local, &result, 1, MPI_DOUBLE, MPI_SUM, comm);
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

void gemv(double alpha, const double *A, const double *x, double beta, double *y, size_t num_rows, size_t num_cols)
{
    // y = alpha * A * x + beta * y;
#pragma omp parallel for
    for (size_t r = 0; r < num_rows; r++)
    {
        double y_val = 0.0;
        for (size_t c = 0; c < num_cols; c++)
        {
            y_val += alpha * A[r * num_cols + c] * x[c];
        }
        y[r] = beta * y[r] + y_val;
    }
}

void conjugate_gradients(const double *A, const double *b, double *x, size_t size, int max_iters, double rel_error, MPI_Comm comm)
{
    int rank, num_procs;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &num_procs);

    double alpha, beta, bb, rr, rr_new;
    double *r = new double[size];
    double *p = new double[size];
    double *Ap = new double[size];
    int num_iters;

    for (size_t i = 0; i < size; i++)
    {
        x[i] = 0.0;
        r[i] = b[i];
        p[i] = b[i];
    }

    bb = dot(b, b, size, comm);
    rr = bb;
    for (num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        gemv(1.0, A, p, 0.0, Ap, size, size);
        alpha = rr / dot(p, Ap, size, comm);
        axpby(alpha, p, 1.0, x, size);
        axpby(-alpha, Ap, 1.0, r, size);
        rr_new = dot(r, r, size, comm);
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
        axpby(1.0, r, beta, p, size);
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

    printf("MPI Implementation, #Processes %d\n", num_procs);

    printf("Usage: ./random_matrix input_file_matrix.bin input_file_rhs.bin output_file_sol.bin max_iters rel_error\n");
    printf("All parameters are optional and have default values\n");
    printf("\n");

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

    double *matrix;
    double *rhs;
    size_t size;

    // Variables for MPI
    size_t matrix_size;
    size_t rows_per_process;
    size_t extra_rows = 0;

    if (rank == 0)
    {

        printf("Reading matrix from file ...\n");
        size_t matrix_rows;
        size_t matrix_cols;
        bool success_read_matrix = read_matrix_from_file(input_file_matrix, &matrix, &matrix_rows, &matrix_cols);
        if (!success_read_matrix)
        {
            fprintf(stderr, "Failed to read matrix\n");
            return 1;
        }

        printf("Process %d -> Read Matrix Done\n", rank);
        printf("\n");

        if (rank == 0)
        {
            printf("Reading right hand side from file ...\n");
        }
        size_t rhs_rows;
        size_t rhs_cols;
        bool success_read_rhs = read_matrix_from_file(input_file_rhs, &rhs, &rhs_rows, &rhs_cols);
        if (!success_read_rhs)
        {
            fprintf(stderr, "Failed to read right hand side\n");
            return 2;
        }
        printf("Process %d -> Reading RHS Done\n", rank);
        printf("\n");

        if (matrix_rows != matrix_cols)
        {
            fprintf(stderr, "Matrix has to be square\n");
            return 3;
        }
        if (rhs_rows != matrix_rows)
        {
            fprintf(stderr, "Size of right hand side does not match the matrix\n");
            return 4;
        }
        if (rhs_cols != 1)
        {
            fprintf(stderr, "Right hand side has to have just a single column\n");
            return 5;
        }

        size = matrix_rows;

        // Matrix size for MPI
        matrix_size = matrix_rows;

        // Dividing by process
        rows_per_process = matrix_rows / num_procs;
        extra_rows = matrix_rows % num_procs;
    }

    // Broadcast matrix_size, rows_per_process, and extra_rows
    MPI_Bcast(&matrix_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&rows_per_process, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&extra_rows, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // Allocate memory for local matrix and RHS vector
    double *local_matrix = new double[(rows_per_process + (rank == 0 ? extra_rows : 0)) * matrix_size];
    double *local_rhs = new double[rows_per_process + (rank == 0 ? extra_rows : 0)];

    // Scatter matrix and RHS vector to all processes
    MPI_Scatter(matrix, (rows_per_process + (rank == 0 ? extra_rows : 0)) * matrix_size, MPI_DOUBLE,
                local_matrix, (rows_per_process + (rank == 0 ? extra_rows : 0)) * matrix_size, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Scatter(rhs, rows_per_process + (rank == 0 ? extra_rows : 0), MPI_DOUBLE,
                local_rhs, rows_per_process + (rank == 0 ? extra_rows : 0), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Solving the system ...\n");
    }

    double *sol = new double[rows_per_process + (rank == 0 ? extra_rows : 0)];
    conjugate_gradients(local_matrix, local_rhs, sol, rows_per_process + (rank == 0 ? extra_rows : 0),
                        max_iters, rel_error, MPI_COMM_WORLD);
    printf("Process %d -> CG Done\n", rank);
    printf("\n");
    double *g_sol = nullptr;
    if (rank == 0)
    {
        g_sol = new double[matrix_size];
    }

    MPI_Gather(sol, rows_per_process + (rank == 0 ? extra_rows : 0), MPI_DOUBLE,
               g_sol, rows_per_process + (rank == 0 ? extra_rows : 0), MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        bool success_write_sol = write_matrix_to_file(output_file_sol, g_sol, matrix_size, 1);
        if (!success_write_sol)
        {
            fprintf(stderr, "Failed to save solution\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // }
        // if (rank == 0)
        // {
        //     printf("Writing solution to file ...\n");

        //     bool success_write_sol = write_matrix_to_file(output_file_sol, sol, size, 1);
        //     if (!success_write_sol)
        //     {
        //         fprintf(stderr, "Failed to save solution\n");
        //         return 6;
        //     }
        //     printf("Done\n");
        //     printf("\n");
    }

    if (rank == 0)
    {
        delete[] matrix;
        delete[] rhs;
    }

    delete[] sol;

    // MPI Cleanup
    delete[] g_sol;
    delete[] local_matrix;
    delete[] local_rhs;

    MPI_Finalize();

    printf("Finished successfully\n");

    return 0;
}