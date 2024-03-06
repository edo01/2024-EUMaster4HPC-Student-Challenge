#ifndef CONJUGATEGRADIENT_GPU_CUDA_CUH
#define CONJUGATEGRADIENT_GPU_CUDA_CUH

#include <cuda.h>
#include <memory>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "../../ConjugateGradient.hpp"

constexpr int NUM_BLOCKS=1000;
constexpr int NUM_THREADS=1024;

namespace LAM
{
    template<typename FloatingType>
    class ConjugateGradient_GPU_CUDA:
    public ConjugateGradient<FloatingType> {
        
        public:
            bool virtual solve(int max_iters, FloatingType rel_error);
            
            bool virtual load_matrix_from_file(const char* filename);
            bool virtual load_rhs_from_file(const char* filename);
            bool virtual save_result_to_file(const char * filename) const;
            
        private:
            FloatingType * A;
            FloatingType * b;
            FloatingType * x;
            size_t size;
    };


}
#endif //CONJUGATEGRADIENT_GPU_CUDA_CUH