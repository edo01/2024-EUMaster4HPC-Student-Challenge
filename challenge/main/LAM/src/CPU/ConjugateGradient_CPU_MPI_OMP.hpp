#ifndef ConjugateGradient_CPU_MPI_OMP_HPP
#define ConjugateGradient_CPU_MPI_OMP_HPP

#include <memory>
#include <iostream>
#include <mpi.h>
#include <chrono>
#include "../ConjugateGradient.hpp"


#define PRINT_RANK0(...) if(rank==0) printf(__VA_ARGS__)
#define PRINT_ERR_RANK0(...) if(rank==0) fprintf(stderr, __VA_ARGS__)

#define FIRST_TOUCH

namespace LAM
{

template<typename FloatingType>
class ConjugateGradient_CPU_MPI_OMP: 
public ConjugateGradient<FloatingType>{
    public:
        using ConjugateGradient<FloatingType>::ConjugateGradient;

        bool virtual solve( int max_iters, FloatingType rel_error);

        bool virtual load_matrix_from_file(const char* filename);
        bool virtual load_rhs_from_file(const char* filename);
        bool virtual save_result_to_file(const char * filename) const;

        bool virtual generate_matrix(const size_t rows, const size_t cols);
        bool virtual generate_rhs();

        size_t get_num_rows() const { return _num_local_rows; }
        size_t get_num_cols() const { return _num_cols; }
    
    private:
        FloatingType* _matrix;
        FloatingType* _rhs;
        FloatingType* _x;
        FloatingType* _r;
        FloatingType* _Ap;
        FloatingType* _p;
    
        size_t _num_local_rows;
        size_t _num_cols;

        // MPI communication variables
        int* _sendcounts;
        int* _displs;
        size_t _offset;
        static MPI_Datatype get_mpi_datatype() {
            if (std::is_same<FloatingType, double>::value) {
                return MPI_DOUBLE;
            } else {
                return MPI_FLOAT;
            }
        }       
        
        FloatingType dot(const FloatingType* x, const FloatingType* y);

        void axpby(FloatingType alpha, const FloatingType* x, FloatingType beta, 
                                FloatingType* y);

        void gemv(FloatingType alpha, const FloatingType* A, const FloatingType* x,
                                FloatingType beta, FloatingType* y);   
        

};

template<typename FloatingType>
bool ConjugateGradient_CPU_MPI_OMP<FloatingType>::solve( int max_iters, FloatingType rel_error)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FloatingType alpha, beta, rhs_module, rr, rr_new;
    int num_iters;

    //first touch policy
    #pragma omp parallel for
    for(size_t i = 0; i < _num_cols ; i++)
    {
        _Ap[i] = 0.0;
        _x[i] = 0.0;
        _r[i] = _rhs[i];
        _p[i] = _rhs[i];
    }

    rhs_module = dot(_rhs, _rhs);

    rr = rhs_module;
    //calculate avarage time of iterations
    std::chrono::duration<double> avg_time(0);
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        gemv(1.0, _matrix, _p, 0.0, _Ap);
        alpha = rr / dot(_p, _Ap);
        axpby(alpha, _p, 1.0, _x);
        axpby(-alpha, _Ap, 1.0, _r);
        rr_new = dot(_r, _r);
        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / rhs_module) < rel_error) { break; }
        axpby(1.0, _r, beta, _p);

        auto end = std::chrono::high_resolution_clock::now();
        avg_time += end - start;
    }
    avg_time /= num_iters;
    PRINT_RANK0("Average time of iterations: %f s\n", avg_time.count());

    if(num_iters <= max_iters)
    {
        PRINT_RANK0("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / rhs_module));
        return true;
    }
    else
    {
        PRINT_RANK0("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / rhs_module));
        return false;
    }
}

template<typename FloatingType>
bool ConjugateGradient_CPU_MPI_OMP<FloatingType>::generate_rhs()
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    size_t rhs_rows = _num_cols;

    _rhs = new FloatingType[rhs_rows];
    
    //first-touch policy: allocate the vector exploiting NUMA to avoid false sharing
#ifdef FIRST_TOUCH
    #pragma omp parallel for 
#endif //FIRST_TOUCH
    for (int i = 0; i < rhs_rows; i++)
    {
        _rhs[i] = 1.0;
    }

    return true;
}

template<typename FloatingType>
bool ConjugateGradient_CPU_MPI_OMP<FloatingType>::generate_matrix(const size_t num_total_rows, const size_t cols)
{
  
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    _num_cols = cols;
    _num_local_rows = num_total_rows / num_procs;
    // calcolate the offset for each rank
    _offset = _num_local_rows * rank;    

    //the last rank will have the remaining rows
    if(rank == num_procs - 1){
        //add the reminder to the last rank
        _num_local_rows += num_total_rows % num_procs; 
    }

    //calculate the displacement and the sendcounts for the scatterv
    _sendcounts = new int[num_procs]; 
    _displs = new int[num_procs];
    

    for(int i=0; i<num_procs; i++){
        _sendcounts[i] = num_total_rows/num_procs;
        _sendcounts[i] += (i==num_procs-1) ? num_total_rows%num_procs : 0;
        _displs[i] = i * (num_total_rows/num_procs);
        //printf("rank %d) sendcounts[%d] = %d, displs[%d] = %d\n", rank, i, _sendcounts[i], i, _displs[i]);
    }

    long unsigned int DATA_SIZE = _num_local_rows * _num_cols * sizeof(FloatingType);
    PRINT_RANK0("Problem size: %lu bytes (%f GB)\n", DATA_SIZE, DATA_SIZE/(1024.0*1024.0*1024.0));
    DATA_SIZE += _num_cols*5*sizeof(FloatingType);
    PRINT_RANK0("I am trying to allocate %lu bytes (%f GB)\n", DATA_SIZE, DATA_SIZE/(1024.0*1024.0*1024.0));
    fflush(stdout);

    // Allocate memory for the local matrix
    _matrix = new FloatingType[_num_local_rows * _num_cols];

    /*
    In a NUMA system, memory pages can be local to a CPU or remote. By default Linux 
    allocates memory in a first-touch policy, meaning the first write access to a memory 
    page determines on which node the page is physically allocated.

    If your malloc is large enough that new memory is requested from the OS (instead of 
    reusing existing heap memory), this first touch will happen in the initialization. 
    Because you use static scheduling for OpenMP, the same thread will use the memory 
    that initialized it. Therefore, unless the thread gets migrated to a different CPU, 
    which is unlikely, the memory will be local.

    If you don't parallelize the initialization, the memory will end up local to the main 
    thread which will be worse for threads that are on different sockets.
    */
    /*
    The same as above also applies to caches. The initialization will put array elements 
    into the cache of the CPU doing it. If the same CPU accesses the memory during the 
    second phase, it will be cache-hot and ready to use.
    */

    // Read the local matrix after first-touch
    /*MPI_File_read(fhandle, _matrix, _num_local_rows * _num_cols, get_mpi_datatype() , MPI_STATUS_IGNORE);*/

    // generate a random spd dense matrix
#ifdef FIRST_TOUCH
    #pragma omp parallel for
#endif  //FIRST_TOUCH
    for (size_t i = 0; i < _num_local_rows; i++) {
        for (size_t j = 0; j < _num_cols; j++) {
            
            if(i+_offset==j-1 || i+_offset==j+1) 
                _matrix[i * _num_cols + j] = 1;
            else if(i+_offset==j)
                _matrix[i * _num_cols + j] = 2;
            else
                _matrix[i * _num_cols + j] = 0;
        }
    }
    
    //initialize x, p, Ap and r
    _x = new FloatingType[_num_cols];
    _p = new FloatingType[_num_cols];
    _Ap = new FloatingType[_num_cols];
    _r = new FloatingType[_num_cols];
    
    return true;
}

template<typename FloatingType>
bool ConjugateGradient_CPU_MPI_OMP<FloatingType>::load_rhs_from_file(const char * filename)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FILE *file = fopen(filename, "rb");
    if (file == nullptr)
    {
        PRINT_ERR_RANK0("Cannot open output file\n");
        return false;
    }

    size_t rhs_rows;
    size_t rhs_cols;

    fread(&rhs_rows, sizeof(size_t), 1, file);
    fread(&rhs_cols, sizeof(size_t), 1, file);
    
    if (rhs_cols != 1)
    {
        PRINT_ERR_RANK0("Right hand side has to have just a single column\n");
        return false;
    }
    if(rhs_rows != _num_cols)
    {
        PRINT_ERR_RANK0("Size of right hand side does not match the matrix\n");
        return false;
    }

    _rhs = new FloatingType[rhs_rows];
    
    //first-touch policy: allocate the vector exploiting NUMA to avoid false sharing
#ifdef FIRST_TOUCH
    #pragma omp parallel for
    for (int i = 0; i < rhs_rows; i++)
    {
        _rhs[i] = 0;
    }
#endif // FIRST_TOUCH

    fread(_rhs, sizeof(FloatingType), rhs_rows, file);

    fclose(file);

    return true;
}

template<typename FloatingType>
bool ConjugateGradient_CPU_MPI_OMP<FloatingType>::load_matrix_from_file(const char * filename)
{
  
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    size_t num_total_rows;
    MPI_Offset file_offset;

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
    MPI_File_read(fhandle, &num_total_rows, 1 ,MPI_UNSIGNED_LONG , MPI_STATUS_IGNORE);
    MPI_File_read(fhandle, &_num_cols, 1 ,MPI_UNSIGNED_LONG , MPI_STATUS_IGNORE);

    // calcolate the offset for each rank
    _num_local_rows = num_total_rows / num_procs;
    _offset = _num_local_rows * rank;
    file_offset = _num_local_rows * sizeof(FloatingType) * rank * _num_cols;
    

    //the last rank will have the remaining rows
    if(rank == num_procs - 1){
        //add the reminder to the last rank
        _num_local_rows += num_total_rows % num_procs; 
    }
    //printf("rank %d) num_total_rows = %lu, num_cols = %lu, num_local_rows = %lu, \
    offset = %lu, file_offset = %lu \n", rank, num_total_rows, _num_cols, _num_local_rows, _offset, file_offset);

    //calculate the displacement and the sendcounts for the scatterv
    _sendcounts = new int[num_procs]; 
    _displs = new int[num_procs];
    

    for(int i=0; i<num_procs; i++){
        _sendcounts[i] = num_total_rows/num_procs;
        _sendcounts[i] += (i==num_procs-1) ? num_total_rows%num_procs : 0;
        _displs[i] = i * (num_total_rows/num_procs);
        //printf("rank %d) sendcounts[%d] = %d, displs[%d] = %d\n", rank, i, _sendcounts[i], i, _displs[i]);
    }

    // seek the file to the correct position for each rank
    MPI_File_seek(fhandle, file_offset, MPI_SEEK_CUR);

    // Allocate memory for the local matrix
    long unsigned int DATA_SIZE = _num_local_rows * _num_cols * sizeof(FloatingType);
    DATA_SIZE += _num_cols*5*sizeof(FloatingType);
    printf("I am trying to allocate %lu bytes (%f GB)\n", DATA_SIZE, DATA_SIZE/(1024*1024*1024));
    fflush(stdout);

    _matrix = new FloatingType[_num_local_rows * _num_cols];

    /*
    In a NUMA system, memory pages can be local to a CPU or remote. By default Linux 
    allocates memory in a first-touch policy, meaning the first write access to a memory 
    page determines on which node the page is physically allocated.

    If your malloc is large enough that new memory is requested from the OS (instead of 
    reusing existing heap memory), this first touch will happen in the initialization. 
    Because you use static scheduling for OpenMP, the same thread will use the memory 
    that initialized it. Therefore, unless the thread gets migrated to a different CPU, 
    which is unlikely, the memory will be local.

    If you don't parallelize the initialization, the memory will end up local to the main 
    thread which will be worse for threads that are on different sockets.
    */
    /*
    The same as above also applies to caches. The initialization will put array elements 
    into the cache of the CPU doing it. If the same CPU accesses the memory during the 
    second phase, it will be cache-hot and ready to use.
    */
#ifdef FIRST_TOUCH
    #pragma omp parallel for
    for(int i=0; i< _num_local_rows * _num_cols; i++){
        _matrix[i]=0;
    }
#endif  //FIRST_TOUCH

    

    // Read the local matrix after first-touch
    MPI_File_read(fhandle, _matrix, _num_local_rows * _num_cols, get_mpi_datatype() , MPI_STATUS_IGNORE);

    MPI_File_close(&fhandle);
    
    //initialize x, p, Ap and r
    _x = new FloatingType[_num_cols];
    _p = new FloatingType[_num_cols];
    _Ap = new FloatingType[_num_cols];
    _r = new FloatingType[_num_cols];
    
    return true;
}

template<typename FloatingType>
bool ConjugateGradient_CPU_MPI_OMP<FloatingType>::save_result_to_file(const char * filename) const
{   
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // only rank 0 will save the result
    if(rank!=0) return true;

    FILE * file = fopen(filename, "wb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }
    int num_cols=1;
    fwrite(&_num_local_rows, sizeof(size_t), 1, file);
    fwrite(&num_cols, sizeof(size_t), 1, file);
    //save rhs to file 
    fwrite(_rhs, sizeof(FloatingType), _num_local_rows, file);

    fclose(file);

    return true;
}

template<typename FloatingType>
FloatingType ConjugateGradient_CPU_MPI_OMP<FloatingType>::dot(const FloatingType* x, const FloatingType* y)
{

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FloatingType result = 0.0;
    FloatingType local_res = 0.0;

    #pragma omp parallel for reduction(+ : local_res)
    for (size_t i = 0; i < _num_local_rows; i++)
    {
        local_res += x[_offset+i] * y[_offset+i];
    }
    
    // Reduce the local results in place
    MPI_Allreduce(&local_res, &result, 1, get_mpi_datatype(), MPI_SUM, MPI_COMM_WORLD);
    
    return result;
}

template<typename FloatingType>
void ConjugateGradient_CPU_MPI_OMP<FloatingType>::axpby(FloatingType alpha, const FloatingType* x,
                                        FloatingType beta, FloatingType* y)
{
    // y = alpha * x + beta * y

    #pragma omp parallel for
    for(size_t i = 0; i < _num_cols; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

template<typename FloatingType>
void ConjugateGradient_CPU_MPI_OMP<FloatingType>::gemv(FloatingType alpha, const FloatingType* A, 
                const FloatingType* x, FloatingType beta, FloatingType* y)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    FloatingType *y_temp = new FloatingType[_num_local_rows];
    FloatingType y_val;

    // y = alpha * A * x + beta * y;
    #pragma omp parallel for
    for(size_t r = 0; r < _num_local_rows; r++)
    {
        FloatingType y_val = 0.0;
        for(size_t c = 0; c < _num_cols; c++)
        {
            y_val += alpha * A[r * _num_cols + c] * x[c];
        }
        y_temp[r] = beta * y[_offset+r] + y_val;
    }

    MPI_Allgatherv(y_temp, _num_local_rows, get_mpi_datatype(), y, _sendcounts, _displs, get_mpi_datatype(), MPI_COMM_WORLD);

    delete[] y_temp;
}

}

#endif //ConjugateGradient_CPU_MPI_OMP_HPP