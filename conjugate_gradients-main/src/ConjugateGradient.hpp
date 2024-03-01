#ifndef CONJUGATEGRADIENT_HPP
#define CONJUGATEGRADIENT_HPP

#include <memory>
#include <iostream>
#include "MelBLAS/include/MelBLAS.hpp"

using namespace melblas;

#define FIRST_TOUCH

template<typename FloatingType>
class ConjugateGradient{
    public:
        ConjugateGradient(std::unique_ptr<MelBLAS_B<FloatingType>>&& melblas): _melblas(std::move(melblas))
        {
            static_assert(std::is_floating_point<FloatingType>::value, "DataType must be floating point");

        };

        ConjugateGradient(std::unique_ptr<MelBLAS_B<FloatingType>>&& melblas, const char* filename): _melblas(std::move(melblas))
        {
            load_matrix_from_file(filename);
            static_assert(std::is_floating_point<FloatingType>::value, "DataType must be floating point");

        };
        
        bool solve( int max_iters, FloatingType rel_error);

        bool load_matrix_from_file(const char* filename);
        bool load_rhs_from_file(const char* filename);
        bool save_result_to_file(const char * filename) const;
        
        size_t get_num_rows() const { return _num_rows; }
        size_t get_num_cols() const { return _num_cols; }

    private:    
        std::unique_ptr<MelBLAS_B<FloatingType>> _melblas;

        FloatingType* _matrix;
        FloatingType* _rhs;
        FloatingType* _x;
        FloatingType* _r;
        FloatingType* _Ap;
        FloatingType* _p;
    
        size_t _num_rows;
        size_t _num_cols;
};

template<typename FloatingType>
bool ConjugateGradient<FloatingType>::solve( int max_iters, FloatingType rel_error)
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

    rhs_module = _melblas->dot(_rhs, _rhs, _num_rows);

    rr = rhs_module;
    for(num_iters = 1; num_iters <= max_iters; num_iters++)
    {
        _melblas->gemv(1.0, _matrix, _p, 0.0, _Ap, _num_rows , _num_cols);
        alpha = rr / _melblas->dot(_p, _Ap, _num_rows);
        _melblas->axpby(alpha, _p, 1.0, _x, _num_rows);
        _melblas->axpby(-alpha, _Ap, 1.0, _r, _num_rows);
        rr_new = _melblas->dot(_r, _r, _num_rows);
        beta = rr_new / rr;
        rr = rr_new;
        if(std::sqrt(rr / rhs_module) < rel_error) { break; }
        _melblas->axpby(1.0, _r, beta, _p, _num_rows);
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
bool ConjugateGradient<FloatingType>::load_rhs_from_file(const char * filename)
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
    
    //first-touch policy: allocate the matrix exploiting NUMA to avoid false sharing
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
bool ConjugateGradient<FloatingType>::load_matrix_from_file(const char * filename)
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
bool ConjugateGradient<FloatingType>::save_result_to_file(const char * filename) const
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



#endif