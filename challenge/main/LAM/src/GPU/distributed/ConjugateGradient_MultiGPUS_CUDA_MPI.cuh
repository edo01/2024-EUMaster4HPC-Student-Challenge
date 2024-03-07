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
#include "../../ConjugateGradient.hpp"

#define PRINT_RANK0(...) if(rank==0) printf(__VA_ARGS__)
#define PRINT_ERR_RANK0(...) if(rank==0) fprintf(stderr, __VA_ARGS__)


namespace LAM
{

    //constexpr int NUM_BLOCKS=1;
    //constexpr int NUM_THREADS=32;

    template<typename FloatingType>
    class ConjugateGradient_MultiGPUS_CUDA_MPI:
    public ConjugateGradient<FloatingType>
    {
        public:
            //using ConjugateGradient<FloatingType>::ConjugateGradient;
            //define a constructor
            ConjugateGradient_MultiGPUS_CUDA_MPI()
            {
                // Initialize CUBLAS
                /*cublas_handler = new cublasHandle_t[_numDevices];
                cublasStatus_t cudaStatus;
                for(int i = 0; i < _numDevices; i++){
                    cudaSetDevice(i);
                    cudaStatus = cublasCreate(&cublas_handler[i]);
                    if (cudaStatus != CUBLAS_STATUS_SUCCESS) {
                        printf("cublasCreate failed\n");
                        fflush(stdout);
                    }else{
                        printf("cublasCreate success\n");
                        fflush(stdout);
                    }
                }*/

            }

            bool virtual solve(int max_iters, FloatingType rel_error);
            
            bool virtual load_matrix_from_file(const char* filename);
            bool virtual load_rhs_from_file(const char* filename);
            bool virtual save_result_to_file(const char * filename) const;
            
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

            //static constexpr ncclDataType_t nccl_datatype = std::is_same<FloatingType, double>::value ? ncclDouble : ncclFloat;

    };
    
}
#endif //ConjugateGradient_MultiGPUS_CUDA_MPI_CUH