#ifndef CONJUGATEGRADIENT_MULTIGPUS_CUDA_CUH
#define CONJUGATEGRADIENT_MULTIGPUS_CUDA_CUH

#include <cuda.h>
#include <memory>
#include <stdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include "../ConjugateGradient.hpp"
#include "ConjugateGradient_CUDA_Kernels.cuh"


constexpr int NUM_BLOCKS 1000;
constexpr int NUM_THREADS 1024;

namespace LAM
{
    template<typename FloatingType>
    class ConjugateGradient_MultiGPUS_CUDA:
    public ConjugateGradient<FloatingType> {
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
            FloatingType * A;
            FloatingType * b;
            FloatingType * x;
            size_t size;
            int numDevices = 1;
    };

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA<FloatingType>::solve( int max_iters, FloatingType rel_error)
    {
        FloatingType * b_dev, * x_dev, * alpha_dev, * beta_dev, * bb_dev, * rr_dev, * rr_new_dev, * r_dev, *pAp_dev, *Ap0_dev;
        FloatingType * rr, * bb;
        rr = new FloatingType;
        bb = new FloatingType;
        int num_iters = 1;

        // Retrieve numbers of devices
        cudaGetDeviceCount(&numDevices);

        // Divide matrices
        size_t ** d_size;
        cudaHostAlloc(&d_size, sizeof(size_t *) * numDevices, cudaHostAllocDefault);
        unsigned int numRowsPerDevice = size / numDevices;
        size_t s = 0;
        for(int i = 0; i < numDevices; i++){
            cudaHostAlloc(&d_size[i], sizeof(size_t), cudaHostAllocDefault);
            *d_size[i] = (s + numRowsPerDevice <= size) ? numRowsPerDevice : size - s;
            s += numRowsPerDevice;
        }
        // if all the rows aren't covered, we add the uncovered row to the last
        if(s < size) *d_size[numDevices - 1] += size - s;

        // Create cuda streams and associate each cuda stream to a device
        cudaStream_t streams[numDevices];
        #pragma omp parallel for num_threads(numDevices)
        for(int i = 0; i < numDevices; i++){
            cudaSetDevice(i);
            cudaStreamCreate(&streams[i]);
        }

        // Allocate and initialize a given number of rows of A to each device
        FloatingType * A_dev[numDevices], * Ap_dev[numDevices], * p_dev[numDevices];
        #pragma omp parallel for num_threads(numDevices)
        for(int i = 0; i < numDevices; i++){
            cudaSetDevice(i);
            cudaMalloc(&A_dev[i], size * (*d_size[i]) * sizeof(FloatingType));
            cudaMalloc(&Ap_dev[i],  (*d_size[i]) * sizeof(FloatingType));
            cudaMalloc(&p_dev[i], size * sizeof(FloatingType));
            unsigned int n = (i == numDevices - 1) ? size - *d_size[i] : numRowsPerDevice * i;
            cudaMemcpyAsync(A_dev[i], A + size * n, size * (*d_size[i]) * sizeof(FloatingType), cudaMemcpyHostToDevice, streams[i]);
        }

        // Allocate in device 0
        cudaSetDevice(0);
        cudaMalloc(&b_dev, sizeof(FloatingType) * size);
        cudaMalloc(&x_dev, sizeof(FloatingType) * size);
        cudaMalloc(&alpha_dev, sizeof(FloatingType));
        cudaMalloc(&beta_dev, sizeof(FloatingType));
        cudaMalloc(&bb_dev, sizeof(FloatingType));
        cudaMalloc(&rr_dev, sizeof(FloatingType));
        cudaMalloc(&rr_new_dev, sizeof(FloatingType));
        cudaMalloc(&r_dev, sizeof(FloatingType) * size);
        cudaMalloc(&pAp_dev, sizeof(FloatingType));
        cudaMalloc(&Ap0_dev, sizeof(FloatingType) * size); // Ap0_dev is located in device 0 and will collect all the result from the devices

        // Initialize variables in device 0
        cudaMemcpyAsync(b_dev, b, sizeof(FloatingType) * size, cudaMemcpyHostToDevice, streams[0]);
        cudaMemsetAsync(x_dev, 0, sizeof(FloatingType) * size, streams[0]); // x = 0
        cudaMemcpyAsync(r_dev, b_dev, sizeof(FloatingType) * size, cudaMemcpyDeviceToDevice, streams[0]); // r = b
        cudaMemcpyAsync(p_dev[0], b_dev, sizeof(FloatingType) * size, cudaMemcpyDeviceToDevice, streams[0]); // p = b

        dot<FloatingType, NUM_BLOCKS, NUM_THREADS>(b_dev ,b_dev, bb_dev, size, streams[0]); // bb = b * b
        cudaMemcpyAsync(rr_dev, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToDevice, streams[0]); // rr = bb


        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {

            // Copy p from device 0 to all other devices to allow matrix vector multiplication
            cudaSetDevice(0);
            #pragma omp parallel for num_threads(numDevices)
            for(int i = 1; i < numDevices; i++){
                cudaError_t error;
                int canPeer;
                cudaSetDevice(i);
                cudaDeviceCanAccessPeer(&canPeer, i, 0);
                if(canPeer == 0) {
                    error = cudaDeviceEnablePeerAccess(0, 0);
                }
                if (error == cudaSuccess || canPeer == 1) {
                    cudaMemcpyPeerAsync(p_dev[i], i, p_dev[0], 0, size * sizeof(FloatingType), streams[i]);
                }
            }

            // Performs matrix vector multiplication in each device
            #pragma omp parallel for num_threads(numDevices)
            for(int i = 0; i < numDevices; i++){
                cudaSetDevice(i);
                gemv_host<FloatingType, NUM_BLOCKS, NUM_THREADS>(1.0, A_dev[i], p_dev[i], 0.0, Ap_dev[i], *d_size[i], size, streams[i]);
            }

            cudaSetDevice(0);
            cudaMemcpyAsync(Ap0_dev, Ap_dev[0], *d_size[0] * sizeof(FloatingType), cudaMemcpyDeviceToDevice, streams[0]);
            #pragma omp parallel for num_threads(numDevices)
            for(int i = 1; i < numDevices; i++){
                cudaError_t error;
                int canPeer;
                cudaSetDevice(0);
                cudaDeviceCanAccessPeer(&canPeer, 0, i);
                if(canPeer == 0) {
                    error = cudaDeviceEnablePeerAccess(i, 0);
                }
                if (error == cudaSuccess || canPeer == 1) {
                    cudaMemcpyPeerAsync(Ap0_dev + i * numRowsPerDevice, 0, Ap_dev[i], i, *d_size[i] * sizeof(FloatingType), streams[0]);
                }
            }

            cudaDeviceSynchronize();

            cudaSetDevice(0);

            dot<FloatingType, NUM_BLOCKS, NUM_THREADS>(p_dev[0], Ap0_dev, pAp_dev, size, streams[0]);

            divide<FloatingType><<<1, 1, 0, streams[0]>>>(rr_dev, pAp_dev, alpha_dev);

            axpy<FloatingType, NUM_BLOCKS, NUM_THREADS><<<NUM_BLOCKS, NUM_THREADS, 0, streams[0]>>>(alpha_dev, p_dev[0], x_dev, size);

            minusaxpy<FloatingType, NUM_BLOCKS, NUM_THREADS><<<NUM_BLOCKS, NUM_THREADS, 0, streams[0]>>>(alpha_dev, Ap0_dev, r_dev, size);

            dot<FloatingType, NUM_BLOCKS, NUM_THREADS>(r_dev, r_dev, rr_new_dev, size, streams[0]);

            divide<FloatingType><<<1, 1, 0, streams[0]>>>(rr_new_dev, rr_dev, beta_dev);

            cudaMemcpyAsync(rr_dev, rr_new_dev, sizeof(FloatingType), cudaMemcpyDeviceToDevice, streams[0]);

            cudaMemcpyAsync(rr, rr_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost, streams[0]);
            cudaMemcpyAsync(bb, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost, streams[0]);
            cudaDeviceSynchronize();
            if(std::sqrt(*rr / *bb) < rel_error) { break; }

            xpby<FloatingType, NUM_BLOCKS, NUM_THREADS><<<NUM_BLOCKS, NUM_THREADS,0, streams[0]>>>(r_dev, beta_dev, p_dev[0], size);
        }


        cudaSetDevice(0);
        cudaMemcpy(rr, rr_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);
        cudaMemcpy(bb, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);

        if(num_iters <= max_iters)
        {
            printf("PARALLEL MULTI-GPUS: Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(*rr / *bb));
        }
        else
        {
            printf("PARALLEL MULTI-GPUS: Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(*rr / *bb));
        }

        // Copy solution in host
        cudaSetDevice(0);
        cudaMemcpyAsync(x, x_dev, size * sizeof(FloatingType), cudaMemcpyDeviceToHost, streams[0]);

        // Free GPU memory
        cudaFreeAsync(alpha_dev, streams[0]);
        cudaFreeAsync(beta_dev, streams[0]);
        cudaFreeAsync(bb_dev, streams[0]);
        cudaFreeAsync(rr_dev, streams[0]);
        cudaFreeAsync(rr_new_dev, streams[0]);
        cudaFreeAsync(r_dev, streams[0]);
        cudaFreeAsync(p_dev, streams[0]);
        cudaFreeAsync(Ap0_dev, streams[0]);
        cudaFreeAsync(pAp_dev, streams[0]);
        #pragma omp parallel for num_threads(numDevices)
        for(int i = 0; i < numDevices; i++){
            cudaSetDevice(i);
            cudaFreeAsync(Ap_dev[i], streams[i]);
            cudaFreeAsync(A_dev[i], streams[i]);
            cudaFreeAsync(p_dev[i], streams[i]);
            cudaStreamDestroy(streams[i]);
        }

        // Free CPU memory
        delete bb;
        delete rr;
        cudaFreeHost(d_size);
        return (num_iters <= max_iters);
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA<FloatingType>::load_matrix_from_file(const char * filename)
    {
        FILE * file = fopen(filename, "rb");
        if(file == nullptr)
        {
            fprintf(stderr, "Cannot open output file\n");
            return false;
        }

        size_t _num_rows, _num_cols;
        fread(&_num_rows, sizeof(size_t), 1, file);
        fread(&_num_cols, sizeof(size_t), 1, file);

        if(_num_rows != _num_cols)
        {
            fprintf(stderr, "Matrix has to be square\n");
            return false;
        }
        size = _num_rows;
        A = new FloatingType[_num_rows * _num_cols];

        fread(A, sizeof(FloatingType), _num_rows * _num_cols, file);

        fclose(file);

        return true;
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA<FloatingType>::load_rhs_from_file(const char * filename)
    {
        FILE * file = fopen(filename, "rb");
        if(file == nullptr)
        {
            fprintf(stderr, "Cannot open output file\n");
            return false;
        }

        size_t rhs_rows, rhs_cols;
        fread(&rhs_rows, sizeof(size_t), 1, file);
        fread(&rhs_cols, sizeof(size_t), 1, file);

        if(rhs_cols != 1){
            fprintf(stderr, "The file does not contain a valid rhs\n");
            return false;
        }
        if(rhs_rows != size)
        {
            fprintf(stderr, "Size of right hand side does not match the matrix\n");
            return false;
        }

        b = new FloatingType[rhs_rows];

        fread(b, sizeof(FloatingType), rhs_rows, file);

        fclose(file);

        return true;
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA<FloatingType>::save_result_to_file(const char * filename) const
    {
        FILE * file = fopen(filename, "wb");
        if(file == nullptr)
        {
            fprintf(stderr, "Cannot open output file\n");
            return false;
        }
        int num_cols = 1;
        fwrite(&size, sizeof(size_t), 1, file);
        fwrite(&num_cols, sizeof(size_t), 1, file);
        //save rhs to file
        fwrite(x, sizeof(FloatingType), size, file);

        fclose(file);

        return true;
    }

}


#endif //CONJUGATEGRADIENT_MULTIGPUS_CUDA_CUH
