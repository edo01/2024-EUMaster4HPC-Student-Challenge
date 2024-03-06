#ifndef CONJUGATEGRADIENT_MULTIGPUS_CUDA_NCCL_CUH
#define CONJUGATEGRADIENT_MULTIGPUS_CUDA_NCCL_CUH

#include <memory>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <nccl.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <omp.h>
#include "../../ConjugateGradient.hpp"

#define PRINT_RANK0(...) if(rank==0) printf(__VA_ARGS__)
#define PRINT_ERR_RANK0(...) if(rank==0) fprintf(stderr, __VA_ARGS__)


namespace LAM
{

    //constexpr int NUM_BLOCKS=1000;
    //constexpr int NUM_THREADS=1024;

    constexpr int NUM_BLOCKS=1;
    constexpr int NUM_THREADS=32;

    template<typename FloatingType>
    class ConjugateGradient_MultiGPUS_CUDA_NCCL:
    public ConjugateGradient<FloatingType>
    {
        public:
            //using ConjugateGradient<FloatingType>::ConjugateGradient;
            //define a constructor
            ConjugateGradient_MultiGPUS_CUDA_NCCL()
            {
                // Retrieve numbers of devices
                cudaGetDeviceCount(&_numDevices);

                // Create cuda streams and associate each cuda stream to a device in the rank
                streams = new cudaStream_t[_numDevices];

                #pragma omp parallel for num_threads(_numDevices)
                for(int i = 0; i < _numDevices; i++){
                    cudaSetDevice(i);
                    cudaStreamCreate(&streams[i]);
                }

                        // Initialize CUBLAS
                cublas_handler = new cublasHandle_t[_numDevices];
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
                }

            }

            bool virtual solve(int max_iters, FloatingType rel_error);
            
            bool virtual load_matrix_from_file(const char* filename);
            bool virtual load_rhs_from_file(const char* filename);
            bool virtual save_result_to_file(const char * filename) const;
            
            size_t get_num_rows() const { return _num_local_rows; }
            size_t get_num_cols() const { return _num_cols; }
            
            //destroy the streams
            ~ConjugateGradient_MultiGPUS_CUDA_NCCL()
            {   
                /*if(_rows_per_device != nullptr)
                    cudaFreeHost(_rows_per_device);
                if(_x != nullptr)
                    cudaFreeHost(_x);
                if(_rhs != nullptr)
                    cudaFreeHost(_rhs);*/
                printf("Destructor of ConjugateGradient_MultiGPUS_CUDA_NCCL\n");
                fflush(stdout);
                for(int i = 0; i < _numDevices; i++)
                    cudaStreamDestroy(streams[i]);
                printf("Streams destroyed\n");
                fflush(stdout);

                //clean _A_dev
                for(int i = 0; i < _numDevices; i++)
                    if(_A_dev[i] != nullptr)
                        cudaFree(_A_dev[i]);
                printf("_A_dev destroyed\n");
                fflush(stdout);
            }

        private:
            const char * filename_matrix;
            const char * filename_rhs;

            // at the index i, it contains the pointer to the portion of matrix in the i-th device
            FloatingType** _A_dev;

            // at the index i, it contains the number of rows of the portion of the matrix that will
            // be transfered to the i-th device
            size_t* _rows_per_device;

            FloatingType * _x;
            FloatingType * _rhs;

            // total number of columns of the matrix
            size_t _num_cols;
            // total number of rows of the local matrix of the rank
            size_t _num_local_rows;
            // offset in the vector of the rank
            size_t _offset;
            
            cudaStream_t *streams;
            cublasHandle_t *cublas_handler;

            //size_t size;
            int _numDevices;

            static MPI_Datatype get_mpi_datatype() {
                if (std::is_same<FloatingType, double>::value) {
                    return MPI_DOUBLE;
                } else {
                    return MPI_FLOAT;
                }
            }

            static constexpr ncclDataType_t nccl_datatype = std::is_same<FloatingType, double>::value ? ncclDouble : ncclFloat;

    };
    
}
#endif //CONJUGATEGRADIENT_MULTIGPUS_CUDA_NCCL_CUH