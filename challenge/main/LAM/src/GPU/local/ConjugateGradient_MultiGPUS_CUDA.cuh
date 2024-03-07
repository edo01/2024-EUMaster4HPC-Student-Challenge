#ifndef CONJUGATEGRADIENT_MULTIGPU_CUDA_CUH
#define CONJUGATEGRADIENT_MULTIGPU_CUDA_CUH

#include <cuda.h>
#include <memory>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "../../ConjugateGradient.hpp"


namespace LAM
{
    template<typename FloatingType>
    class ConjugateGradient_MultiGPUS_CUDA:
    public ConjugateGradient<FloatingType> {
        
        public:
            ConjugateGradient_MultiGPUS_CUDA(){
                cudaGetDeviceCount(&numDevices);
            }

            bool virtual solve(int max_iters, FloatingType rel_error);
            
            bool virtual load_matrix_from_file(const char* filename);
            bool virtual load_rhs_from_file(const char* filename);
            bool virtual save_result_to_file(const char * filename) const;
            
        private:
            FloatingType * A;
            FloatingType * b;
            FloatingType * x;
            size_t size;
            int numDevices;
    };


}
#endif //CONJUGATEGRADIENT_MULTIGPU_CUDA_CUH