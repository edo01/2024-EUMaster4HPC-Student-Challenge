#ifndef ConjugateGradient_MultiGPUS_CUDA_MPI_CUH
#define ConjugateGradient_MultiGPUS_CUDA_MPI_CUH

#include <memory>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include <stdint.h>
#include <mpi.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>
#include "../../ConjugateGradient.hpp"



namespace LAM
{

    template<typename FloatingType>
    class ConjugateGradient_MultiGPUS_CUDA_MPI:
    public ConjugateGradient<FloatingType>
    {
        public:
            using ConjugateGradient<FloatingType>::ConjugateGradient;
            //define a constructor

            bool virtual solve(int max_iters, FloatingType rel_error);
            
            bool virtual load_matrix_from_file(const char* filename);
            bool virtual load_rhs_from_file(const char* filename);
            bool virtual save_result_to_file(const char * filename) const;

            bool virtual generate_matrix(const size_t rows, const size_t cols);
            bool virtual generate_rhs();
            
            size_t get_num_rows() const { return _num_local_rows; }
            size_t get_num_cols() const { return _num_cols; }
            
            //destroy the streams
            ~ConjugateGradient_MultiGPUS_CUDA_MPI()
            {   
                if(_x != nullptr)
                    cudaFreeHost(_x);
                if(_rhs != nullptr)
                    cudaFreeHost(_rhs);

                cudaStreamDestroy(stream);

                //clean _A_dev
                if(_A_dev != nullptr)
                    cudaFree(_A_dev);
                
            }

        private:
            const char * filename_matrix;
            const char * filename_rhs;

            FloatingType* _A_dev;
            FloatingType * _x;
            FloatingType * _rhs;

            // total number of columns of the matrix
            size_t _num_cols;
            // total number of rows of the local matrix of the rank
            size_t _num_local_rows;

            int _device_id;
            
            // MPI communication variables
            int* _sendcounts;
            int* _displs;

            cudaStream_t stream;

            static MPI_Datatype get_mpi_datatype() {
                if (std::is_same<FloatingType, double>::value) {
                    return MPI_DOUBLE;
                } else {
                    return MPI_FLOAT;
                }
            }

    };
    
}
#endif //ConjugateGradient_MultiGPUS_CUDA_MPI_CUH