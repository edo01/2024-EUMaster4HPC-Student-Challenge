#ifndef CONJUGATEGRADIENT_GPU_CUDA_CUH
#define CONJUGATEGRADIENT_GPU_CUDA_CUH

#include <cuda.h>
#include <memory>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include "ConjugateGradient.hpp"
#include "ConjugateGradient_CUDA_Kernels.cuh"

constexpr int NUM_BLOCKS 1000;
constexpr int NUM_THREADS 1024;

namespace LAM
{
    template<typename FloatingType>
    class ConjugateGradient_GPU_CUDA:
            public ConjugateGradient<FloatingType> {

        private:
            FloatingType * A;
            FloatingType * b;
            FloatingType * x;
            size_t size;
    };


    template<typename FloatingType>
    bool ConjugateGradient_GPU_CUDA<FloatingType>::solve( int max_iters, FloatingType rel_error) {
        FloatingType * A_dev, * b_dev, * x_dev, * alpha_dev, * beta_dev, * bb_dev, * rr_dev, * rr_new_dev, * r_dev, * p_dev, * Ap_dev, * pAp_dev;
        FloatingType * rr, * bb;
        rr = new FloatingType;
        bb = new FloatingType;
        int num_iters;

        // Allocate memory in GPU
        cudaMalloc(&A_dev, sizeof(FloatingType) * size * size);
        cudaMalloc(&b_dev, sizeof(FloatingType) * size);
        cudaMalloc(&x_dev, sizeof(FloatingType) * size);
        cudaMemcpy(A_dev, A, sizeof(FloatingType) * size * size, cudaMemcpyHostToDevice);
        cudaMemcpy(b_dev, b, sizeof(FloatingType) * size, cudaMemcpyHostToDevice);

        // Allocate memory in GPU
        cudaMalloc(&alpha_dev, sizeof(FloatingType));
        cudaMalloc(&beta_dev, sizeof(FloatingType));
        cudaMalloc(&bb_dev, sizeof(FloatingType));
        cudaMalloc(&rr_dev, sizeof(FloatingType));
        cudaMalloc(&rr_new_dev, sizeof(FloatingType));
        cudaMalloc(&r_dev, sizeof(FloatingType) * size);
        cudaMalloc(&p_dev, sizeof(FloatingType) * size);
        cudaMalloc(&Ap_dev, sizeof(FloatingType) * size);
        cudaMalloc(&pAp_dev, sizeof(FloatingType));

        // Initialize variables in GPU
        cudaMemset(x_dev, 0, sizeof(FloatingType) * size); // x = 0
        cudaMemcpy(r_dev, b_dev, sizeof(FloatingType) * size, cudaMemcpyDeviceToDevice); // r = b
        cudaMemcpy(p_dev, b_dev, sizeof(FloatingType) * size, cudaMemcpyDeviceToDevice); // p = b

        dot<FloatingType, NUM_BLOCKS, NUM_THREADS>(b_dev ,b_dev, bb_dev, size); // bb = b * b
        cudaMemcpy(rr_dev, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToDevice); // rr = bb

        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {

            gemv_host<FloatingType, NUM_BLOCKS, NUM_THREADS>(1.0, A_dev, p_dev, 0.0, Ap_dev, size, size);

            dot<FloatingType, NUM_BLOCKS, NUM_THREADS>(p_dev, Ap_dev, pAp_dev, size);

            divide<FloatingType><<<1, 1>>>(rr_dev, pAp_dev, alpha_dev);

            axpy<FloatingType, NUM_BLOCKS, NUM_THREADS><<<NUM_BLOCKS, NUM_THREADS>>>(alpha_dev, p_dev, x_dev, size);

            minusaxpy<FloatingType, NUM_BLOCKS, NUM_THREADS><<<NUM_BLOCKS, NUM_THREADS>>>(alpha_dev, Ap_dev, r_dev, size);

            dot<FloatingType, NUM_BLOCKS, NUM_THREADS>(r_dev, r_dev, rr_new_dev, size);

            divide<FloatingType><<<1, 1>>>(rr_new_dev, rr_dev, beta_dev);

            cudaMemcpy(rr_dev, rr_new_dev, sizeof(FloatingType), cudaMemcpyDeviceToDevice);

            cudaMemcpy(rr, rr_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);
            cudaMemcpy(bb, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);
            if(std::sqrt(*rr / *bb) < rel_error) { break; }

            xpby<FloatingType, NUM_BLOCKS, NUM_THREADS><<<NUM_BLOCKS, NUM_THREADS>>>(r_dev, beta_dev, p_dev, size);
        }

        cudaMemcpy(rr, rr_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);
        cudaMemcpy(bb, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);

        // Copy solution in CPU
        cudaMemcpy(x, x_dev, sizeof(FloatingType) * size, cudaMemcpyDeviceToHost);

        // Free GPU memory
        cudaFree(alpha_dev);
        cudaFree(beta_dev);
        cudaFree(bb_dev);
        cudaFree(rr_dev);
        cudaFree(rr_new_dev);
        cudaFree(r_dev);
        cudaFree(p_dev);
        cudaFree(Ap_dev);
        cudaFree(pAp_dev);
        cudaFree(A_dev);
        cudaFree(b_dev);
        cudaFree(x_dev);

        if(num_iters <= max_iters)
        {
            printf("PARALLEL GPU CUDA: Converged in %d iterations, relative error is %e\n", num_iters, std::sqrt(*rr / *bb));
        }
        else
        {
            printf("PARALLEL GPU CUDA: Did not converge in %d iterations, relative error is %e\n", max_iters, std::sqrt(*rr / *bb));
        }
        // Free CPU memory
        delete bb;
        delete rr;

        return (num_iters <= max_iters);
    }


    template<typename FloatingType>
    bool ConjugateGradient_GPU_CUDA<FloatingType>::load_matrix_from_file(const char * filename) {
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
    bool ConjugateGradient_GPU_CUDA<FloatingType>::load_rhs_from_file(const char * filename) {
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
    bool ConjugateGradient_CPU_OMP<FloatingType>::save_result_to_file(const char * filename) const
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
#endif //CONJUGATEGRADIENT_GPU_CUDA_CUH