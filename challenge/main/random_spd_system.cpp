#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>

#include <mkl.h>



void print_matrix(const double * matrix, size_t num_rows, size_t num_cols, FILE * file = stdout)
{
    fprintf(file, "%zu %zu\n", num_rows, num_cols);
    for(size_t r = 0; r < num_rows; r++)
    {
        for(size_t c = 0; c < num_cols; c++)
        {
            double val = matrix[r * num_cols + c];
            printf("%+6.3f ", val);
        }
        printf("\n");
    }
}



void random_matrix(double * matrix, size_t num_rows, size_t num_cols, int seed)
{
    srand(seed);
    for(size_t c = 0; c < num_cols; c++)
    {
        for(size_t r = 0; r < num_rows; r++)
        {
            matrix[c * num_rows + r] = ((2.0 * rand()) / RAND_MAX) - 1.0;
        }
    }
}



void gram_schmidt_recursive(double * A, size_t col_begin, size_t col_end, size_t num_rows, double * buffer_alphas)
{
    size_t ld = num_rows;
    size_t num_cols_curr = col_end - col_begin;
    if(num_cols_curr == 1)
    {
        double norm = cblas_dnrm2(num_rows, A + col_begin * ld, 1);
        cblas_dscal(num_rows, 1.0/norm, A + col_begin * ld, 1);
        return;
    }

    size_t col_mid = (col_end + col_begin) / 2;
    size_t num_cols_first_half = col_mid - col_begin;
    size_t num_cols_second_half = col_end - col_mid;

    gram_schmidt_recursive(A, col_begin, col_mid, num_rows, buffer_alphas);

    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, num_cols_first_half, num_cols_second_half, num_rows, 1.0, A + col_begin * ld, ld, A + col_mid * ld, ld, 0.0, buffer_alphas, ld);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, num_rows, num_cols_second_half, num_cols_first_half, -1.0, A + col_begin * ld, ld, buffer_alphas, ld, 1.0, A + col_mid * ld, ld);

    gram_schmidt_recursive(A, col_mid, col_end, num_rows, buffer_alphas);
}



void random_spd_matrix(double * A, size_t size, int seed)
{
    // generate random orthogonal matrix Q and diagonal matrix with positive eigenvalues D
    // then A = Q*D*Qt = Q*d*d*Qt = (Q*d)*(dt*Qt) = (Q*d)*(Q*d)^T
    // we are doing the opposite of eigendecomposition of an SPD matrix

    double * Q = new double[size * size];
    double * D = new double[size];
    double * buffer_alphas = new double[size * size];

    // generate random matrix
    random_matrix(Q, size, size, seed);

    // orthonormalize the matrix columns using gram-schmidt
    gram_schmidt_recursive(Q, 0, size, size, buffer_alphas);

    // generate random positive eigenvalues
    random_matrix(D, size, 1, seed - 10);
    for(size_t i = 0; i < size; i++)
    {
        D[i] = std::exp(3.5 * D[i]);
    }

    // multiply Q*d
    for(size_t c = 0; c < size; c++)
    {
        cblas_dscal(size, std::sqrt(D[c]), Q + c * size, 1);
    }

    // multiply (Q*d)*(Q*d)^T
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, size, size, size, 1.0, Q, size, Q, size, 0.0, A, size);

    delete[] Q;
    delete[] D;
    delete[] buffer_alphas;
}



bool write_matrix_to_file(const char * filename, const double * matrix, size_t num_rows, size_t num_cols)
{
    FILE * file = fopen(filename, "wb");
    if(file == nullptr)
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





int main(int argc, char ** argv)
{
    printf("Usage: ./random_spd_system matrix_size output_file_matrix.bin output_file_rhs.bin random_seed\n");
    printf("All parameters are optional and have default values\n");
    printf("\n");

    const char * output_file_matrix = "io/matrix.bin";
    const char * output_file_rhs = "io/rhs.bin";
    size_t size = 10;
    int seed = time(nullptr);

    if(argc > 1) size = static_cast<size_t>(atoll(argv[1]));
    if(argc > 2) output_file_matrix = argv[2];
    if(argc > 3) output_file_rhs = argv[3];
    if(argc > 4) seed = atoi(argv[4]);

    printf("Command line arguments:\n");
    printf("  matrix_size:        %zu\n", size);
    printf("  output_file_matrix: %s\n", output_file_matrix);
    printf("  output_file_rhs:    %s\n", output_file_rhs);
    printf("  seed:               %d\n", seed);
    printf("\n");

    if((ssize_t)size <= 0)
    {
        fprintf(stderr, "Wrong argument value\n");
        return 1;
    }



    printf("Generating the matrix ...\n");
    double * matrix = new double[size * size];
    random_spd_matrix(matrix, size, seed);
    printf("Done\n");
    printf("\n");

    printf("Generating the right hand side ...\n");
    double * rhs = new double[size];
    random_matrix(rhs, size, 1, seed+10);
    printf("Done\n");
    printf("\n");

    printf("Writing matrix to file ...\n");
    bool success_write_matrix = write_matrix_to_file(output_file_matrix, matrix, size, size);
    if(!success_write_matrix)
    {
        fprintf(stderr, "Failed to save matrix\n");
        return 2;
    }
    printf("Done\n");
    printf("\n");

    printf("Writing right hand side to file ...\n");
    bool success_write_rhs = write_matrix_to_file(output_file_rhs, rhs, size, 1);
    if(!success_write_rhs)
    {
        fprintf(stderr, "Failed to save right hand side\n");
        return 3;
    }
    printf("Done\n");
    printf("\n");

    delete[] matrix;
    delete[] rhs;

    printf("Finished successfully\n");

    return 0;
}
