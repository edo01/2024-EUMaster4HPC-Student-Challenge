#ifndef ConjugateGradient_CPU_OMP_HPP
#define ConjugateGradient_CPU_OMP_HPP

#include <memory>
#include <iostream>
#include "ConjugateGradient.hpp"

#define FIRST_TOUCH

namespace LAM
{

template<typename FloatingType>
class ConjugateGradient_CPU_OMP: 
public ConjugateGradient<FloatingType>{
    public:
        using ConjugateGradient<FloatingType>::ConjugateGradient;

        bool virtual solve( int max_iters, FloatingType rel_error);

        bool virtual load_matrix_from_file(const char* filename);
        bool virtual load_rhs_from_file(const char* filename);
        bool virtual save_result_to_file(const char * filename) const;
        
        size_t get_num_rows() const { return _num_rows; }
        size_t get_num_cols() const { return _num_cols; }
    
    private:
        FloatingType* _matrix;
        FloatingType* _rhs;
        FloatingType* _x;
        FloatingType* _r;
        FloatingType* _Ap;
        FloatingType* _p;
    
        size_t _num_rows;
        size_t _num_cols;

        FloatingType dot(const FloatingType* x, const FloatingType* y, size_t size);

        void axpby(FloatingType alpha, const FloatingType* x, FloatingType beta, 
                                FloatingType* y, size_t size);

        void gemv(FloatingType alpha, const FloatingType* A, const FloatingType* x,
                                FloatingType beta, FloatingType* y, size_t rows, size_t cols);   

};

template<typename FloatingType>
bool ConjugateGradient_CPU_OMP<FloatingType>::solve( int max_iters, FloatingType rel_error)
{
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

    rhs_module = dot(_rhs, _rhs, _num_rows);

    rr = rhs_module;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        gemv(1.0, _matrix, _p, 0.0, _Ap, _num_rows , _num_cols);
        alpha = rr / dot(_p, _Ap, _num_rows);
        axpby(alpha, _p, 1.0, _x, _num_rows);
        axpby(-alpha, _Ap, 1.0, _r, _num_rows);
        rr_new = dot(_r, _r, _num_rows);
        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / rhs_module) < rel_error) { break; }
        axpby(1.0, _r, beta, _p, _num_rows);
    }

    if(num_iters <= max_iters)
    {
        printf("Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(rr / rhs_module));
        return true;
    }
    else
    {
        printf("Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(rr / rhs_module));
        return false;
    }
}

template<typename FloatingType>
bool ConjugateGradient_CPU_OMP<FloatingType>::load_rhs_from_file(const char * filename)
{
  
    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    size_t rhs_rows;
    size_t rhs_cols;

    fread(&rhs_rows, sizeof(size_t), 1, file);
    fread(&rhs_cols, sizeof(size_t), 1, file);

    if(rhs_cols != 1){
        fprintf(stderr, "The file does not contain a valid rhs\n");
        return false;
    }
    if(rhs_rows != _num_rows)
    {
        fprintf(stderr, "Size of right hand side does not match the matrix\n");
        return false;
    }

    _rhs = new FloatingType[rhs_rows];
    
    //first-touch policy: allocate the vector exploiting NUMA to avoid false sharing
#ifdef FIRST_TOUCH
    #pragma omp parallel for
    for(int i=0; i<rhs_rows; i++){
        _rhs[i]=0;
    }
#endif //FIRST_TOUCH

    fread(_rhs, sizeof(FloatingType), rhs_rows, file);

    fclose(file);

    return true;
}

template<typename FloatingType>
bool ConjugateGradient_CPU_OMP<FloatingType>::load_matrix_from_file(const char * filename)
{
  
    FILE * file = fopen(filename, "rb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }

    fread(&_num_rows, sizeof(size_t), 1, file);
    fread(&_num_cols, sizeof(size_t), 1, file);
    
    if(_num_rows != _num_cols)
    {
        fprintf(stderr, "Matrix has to be square\n");
        return false;
    }

    _matrix = new FloatingType[_num_rows * _num_cols];
    
    //first-touch policy: allocate the matrix exploiting NUMA to avoid false sharing
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
    for(int i=0; i<_num_rows * _num_cols; i++){
        _matrix[i]=0;
    }
#endif  //FIRST_TOUCH

    //initialize x, p, Ap and r
    _x = new FloatingType[_num_rows];
    _p = new FloatingType[_num_rows];
    _Ap = new FloatingType[_num_rows];
    _r = new FloatingType[_num_rows];

    fread(_matrix, sizeof(FloatingType), _num_rows * _num_cols, file);

    fclose(file);

    return true;
}

template<typename FloatingType>
bool ConjugateGradient_CPU_OMP<FloatingType>::save_result_to_file(const char * filename) const
{
    FILE * file = fopen(filename, "wb");
    if(file == nullptr)
    {
        fprintf(stderr, "Cannot open output file\n");
        return false;
    }
    int num_cols=1;
    fwrite(&_num_rows, sizeof(size_t), 1, file);
    fwrite(&num_cols, sizeof(size_t), 1, file);
    //save rhs to file 
    fwrite(_rhs, sizeof(FloatingType), _num_rows, file);

    fclose(file);

    return true;
}

template<typename FloatingType>
FloatingType ConjugateGradient_CPU_OMP<FloatingType>::dot(const FloatingType* x, const FloatingType* y,
                                                size_t size)
{
    FloatingType result = 0.0;

    #pragma omp parallel for reduction(+:result)
    for(size_t i = 0; i < size; i++)
    {
        result += x[i] * y[i];
    }
    return result;
}

template<typename FloatingType>
void ConjugateGradient_CPU_OMP<FloatingType>::axpby(FloatingType alpha, const FloatingType* x,
                                        FloatingType beta, FloatingType* y, size_t size)
{
    // y = alpha * x + beta * y

    #pragma omp parallel for
    for(size_t i = 0; i < size; i++)
    {
        y[i] = alpha * x[i] + beta * y[i];
    }
}

template<typename FloatingType>
void ConjugateGradient_CPU_OMP<FloatingType>::gemv(FloatingType alpha, const FloatingType* A, 
                const FloatingType* x, FloatingType beta, FloatingType* y,
                size_t rows, size_t cols)
{

    // y = alpha * A * x + beta * y;
    #pragma omp parallel for
    for(size_t r = 0; r < rows; r++)
    {
        FloatingType y_val = 0.0;
        for(size_t c = 0; c < cols; c++)
        {
            y_val += alpha * A[r * cols + c] * x[c];
        }
        y[r] = beta * y[r] + y_val;
    }
}

}

#endif //ConjugateGradient_CPU_OMP_HPP