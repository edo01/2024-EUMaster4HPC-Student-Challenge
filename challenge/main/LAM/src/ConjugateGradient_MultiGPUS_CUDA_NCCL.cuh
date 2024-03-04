#ifndef CONJUGATEGRADIENT_MULTIGPUS_CUDA_NCCL_CUH
#define CONJUGATEGRADIENT_MULTIGPUS_CUDA_NCCL_CUH

#include <cuda.h>
#include <memory>
#include <stdlib>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include "ConjugateGradient.hpp"
#include "ConjugateGradient_CUDA_Kernels.cuh"

constexpr int NUM_BLOCKS 1000;
constexpr int NUM_THREADS 1024;

namespace LAM
{
    template<typename FloatingType>
    class ConjugateGradient_MultiGPUS_CUDA_NCCL:
            public ConjugateGradient<FloatingType>
    {
        public:
            ConjugateGradient_MultiGPUS_CUDA_NCCL(const char * _filename_matrix, onst char * _filename_rhs)
                        : ConjugateGradient<FloatingType>()
            {
                filename_matrix = _filename_matrix;
                filename_rhs = _filename_rhs;
            }


        private:
            const char * filename_matrix;
            const char * filename_rhs;

            FloatingType * x;
            FloatingType * b;

            size_t size;
            int numDevices = 1;


            static constexpr MPI_Datatype mpi_datatype = std::is_same<FloatingType, double>::value ? MPI_DOUBLE : MPI_FLOAT;
            static constexpr NCCL_Datatype nccl_datatype = std::is_same<FloatingType, double>::value ? ncclDouble : ncclFloat;

            void read_and_divide_matrix_from_file(const char * filename, FloatingType ** A_dev, size_t * h_sizes, size_t * num_rows,
                                         cudaStream_t * streams, size_t numDevices, int myRank = 0, int nRanks = 0);
    };

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_NCCL<FloatingType>::solve( int max_iters, FloatingType rel_error)
    {
        FloatingType * b_dev, * x_dev, * alpha_dev, * beta_dev, * bb_dev, * rr_dev, * rr_new_dev, * r_dev, * pAp_dev, * Ap0_dev;
        FloatingType * rr, * bb;
        bool stop = false;
        rr = new FloatingType;
        bb = new FloatingType;
        int num_iters = 1;
        int myRank, nRanks;
        size_t size;

        //initializing MPI
        MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
        MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

        // Retrieve numbers of devices
        cudaGetDeviceCount(&numDevices);

        // Create cuda streams and associate each cuda stream to a device in the rank
        cudaStream_t streams[numDevices];
        #pragma omp parallel for num_threads(numDevices)
        for(int i = 0; i < numDevices; i++){
            cudaSetDevice(i);
            cudaStreamCreate(&streams[i]);
        }

        // Allocate and initialize a given number of rows of A to each device
        FloatingType * A_dev[numDevices], * Ap_dev[numDevices], * p_dev[numDevices];
        size_t * d_size;
        // Read and allocate matrix in GPUs from file
        cudaHostAlloc(&d_size, sizeof(size_t) * numDevices, cudaHostAllocDefault);
        read_and_divide_matrix_from_file<FloatingType>(filename_matrix, A_dev, d_size, &size, streams, numDevices, myRank, nRanks);

        // Allocate p and Ap in each device of each rank
        #pragma omp parallel for num_threads(numDevices)
        for(int i = 0; i < numDevices; i++){
            cudaSetDevice(i);
            cudaMalloc(&Ap_dev[i],  (d_size[i]) * sizeof(FloatingType));
            cudaMalloc(&p_dev[i], size * sizeof(FloatingType));
        }

        // Allocate in device 0 of rank 0 all the needed memory space
        if (myRank == 0) {
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

            //  Only rank 0 reads the right-hand side vector from the matrix
            load_rhs_from_file(filename_rhs); //TODO check if this is right, how to generalize this

            // Initialize variables in device 0
            cudaMemcpyAsync(b_dev, b, sizeof(FloatingType) * size, cudaMemcpyHostToDevice, streams[0]);
            cudaMemsetAsync(x_dev, 0, sizeof(FloatingType) * size, streams[0]); // x = 0
            cudaMemcpyAsync(r_dev, b_dev, sizeof(FloatingType) * size, cudaMemcpyDeviceToDevice, streams[0]); // r = b
            cudaMemcpyAsync(p_dev[0], b_dev, sizeof(FloatingType) * size, cudaMemcpyDeviceToDevice, streams[0]); // p = b

            dot<FloatingType, NUM_BLOCKS, NUM_THREADS>(b_dev, b_dev, bb_dev, size, streams[0]); // bb = b * b
            cudaMemcpyAsync(rr_dev, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToDevice, streams[0]); // rr = bb
        }

        ncclUniqueId id;
        ncclComm_t comms[numDevices];
        // Generating NCCL unique ID at one process and broadcasting it to all
        if (myRank == 0) ncclGetUniqueId(&id);
        MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        //  Initializing NCCL
        ncclGroupStart();
        for (int i=0; i<numDevices; i++) {
            cudaSetDevice(i);
            ncclCommInitRank(comms + i, nRanks * numDevices, id, myRank * numDevices + i);
        }
        ncclGroupEnd();

        // CG Iterations
        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {
            // Copy p from device 0 to all other devices to allow matrix-vector multiplication
            ncclGroupStart();
            for (int i=0; i < numDevices; i++) {
                ncclBroadcast(p_dev[0], p_dev[i], size, nccl_datatype, 0, comms[i], streams[i]);
            }
            ncclGroupEnd();

            // Performs matrix-vector multiplication in each device of each rank
            #pragma omp parallel for num_threads(numDevices)
            for(int i = 0; i < numDevices; i++){
                cudaSetDevice(i);
                gemv_host<FloatingType, NUM_BLOCKS, NUM_THREADS>(1.0, A_dev[i], p_dev[i], 0.0, Ap_dev[i], d_size[i], size, streams[i]);
            }

            // All-To-One Gather to collect all the results of the mat-vec multiplication in device 0 in rank 0
            ncclGroupStart();
            if(myRank == 0) {
                int offset = 0;
                for(int i = 0; i < numDevices * nRanks; i++){
                    if(i < numDevices * (nRanks - 1)) {
                        ncclRecv(Ap0_dev + offset, d_size[i % numDevices], nccl_datatype, i, comms[0], streams[0]);
                        offset += d_size[i % numDevices];
                    } else {
                        unsigned int numRowsLastRank = size / nRanks + size % nRanks;
                        unsigned int numRowsDeviceLastRank = numRowsLastRank / numDevices;
                        if (i == numDevices * nRanks - 1){
                            numRowsDeviceLastRank += numRowsLastRank % numDevices;
                        }
                        ncclRecv(Ap0_dev + offset, numRowsDeviceLastRank, nccl_datatype, i, comms[0], streams[0]);
                        offset += numRowsDeviceLastRank;
                    }
                }
            }
            for(int i = 0; i < numDevices; i++) {
                ncclSend(Ap_dev[i], d_size[i] , nccl_datatype, 0, comms[i], streams[i]);
            }
            ncclGroupEnd();

            // Synchronizing on CUDA stream to complete NCCL communication
            for (int i = 0; i < numDevices; i++) {
                cudaStreamSynchronize(streams[i]);
            }

            // Device 0 in rank 0 carries on all the other operation involved in the iteration of the CG method
            if(myRank == 0) {
                cudaSetDevice(0);

                dot<FloatingType,NUM_BLOCKS, NUM_THREADS>(p_dev[0], Ap0_dev, pAp_dev, size, streams[0]);

                divide<FloatingType><<<1, 1, 0, streams[0]>>>(rr_dev, pAp_dev, alpha_dev);

                axpy<FloatingType, NUM_BLOCKS, NUM_THREADS><<<NUM_BLOCKS, NUM_THREADS, 0, streams[0]>>>
                        (alpha_dev, p_dev[0], x_dev, size);

                minusaxpy<FloatingType, NUM_BLOCKS, NUM_THREADS><<<NUM_BLOCKS, NUM_THREADS, 0, streams[0]>>>
                        (alpha_dev, Ap0_dev, r_dev, size);

                dot<FloatingType, NUM_BLOCKS, NUM_THREADS>(r_dev, r_dev, rr_new_dev, size, streams[0]);

                divide<FloatingType><<<1, 1, 0, streams[0]>>>(rr_new_dev, rr_dev, beta_dev);

                cudaMemcpyAsync(rr_dev, rr_new_dev, sizeof(FloatingType), cudaMemcpyDeviceToDevice, streams[0]);

                cudaMemcpyAsync(rr, rr_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost, streams[0]);
                cudaMemcpyAsync(bb, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost, streams[0]);
                if (std::sqrt(*rr / *bb) < rel_error) { stop = true; }
            }

            // Rank 0 broadcasts the flag stop to all other rank in order to stop the computation when the solution is found
            MPI_Bcast(&stop, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
            if(stop) { break; }

            // Device 0 in rank 0 computes the new value of p that will be broadcast to all other devices in the next iteration
            if (myRank == 0){
                cudaSetDevice(0);
                xpby<FloatingType, NUM_BLOCKS, NUM_THREADS><<<NUM_BLOCKS, NUM_THREADS, 0, streams[0]>>>(r_dev, beta_dev, p_dev[0], size);
            }
        }

        // Device 0 of rank 0 prints the information about the result of the CG method
        if(myRank == 0) {
            cudaSetDevice(0);
            cudaMemcpy(rr, rr_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);
            cudaMemcpy(bb, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);
            // Prints the number of iterations and the relative error
            if (num_iters <= max_iters) {
                printf("PARALLEL MULTI-GPUS CUDA NCCL: Converged in %d iterations, relative error is %e\n", num_iters,
                       std::sqrt(*rr / *bb));
            } else {
                printf("PARALLEL MULTI-GPUS CUDA NCCL: Did not converge in %d iterations, relative error is %e\n", max_iters,
                       std::sqrt(*rr / *bb));
            }
            // Copy solution in host
            cudaSetDevice(0);
            cudaFreeHost(x);
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

            // Free CPU memory
            delete bb;
            delete rr;
            cudaFreeHost(d_size);
        }

        // All devices free their allocated memory and destroy streams
        #pragma omp parallel for num_threads(numDevices)
        for(int i = 0; i < numDevices; i++){
            cudaSetDevice(i);
            cudaFreeAsync(Ap_dev[i], streams[i]);
            cudaFreeAsync(A_dev[i], streams[i]);
            cudaFreeAsync(p_dev[i], streams[i]);
            cudaStreamDestroy(streams[i]);
        }
        // All ranks free host memory
        cudaFreeHost(d_size);

        // Finalizing NCCL
        for (int i = 0; i < numDevices; i++) {
            ncclCommDestroy(comms[i]);
        }

        // Finalizing MPI
        MPI_Finalize();
        printf("[MPI Rank %d] Success \n", myRank);

        return (num_iters <= max_iters);
    }


    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_NCCL<FloatingType>::load_matrix_from_file(const char * filename)
    {

    }


    //  Only rank 0 reads the right-hand side vector from the matrix
    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_NCCL<FloatingType>::load_rhs_from_file(const char * filename)
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


    template<typename FloatingType>
    void ConjugateGradient_MultiGPUS_CUDA<FloatingType>::read_and_divide_matrix_from_file
            (const char * filename, FloatingType ** A_dev, size_t * h_sizes, size_t * num_rows,
             cudaStream_t * streams, size_t numDevices, int myRank = 0, int nRanks = 0)
    {
        // Initialize an MPI file handler and try to open the file
        MPI_File fhandle;
        if(MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandle) != MPI_SUCCESS) {
            printf("[MPI process %d] Failure in opening the file.\n", myRank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        printf("[MPI process %d] File opened successfully.\n", myRank);

        // Read from file the dimensions of the matrix
        size_t * num_cols;
        num_cols = new size_t;
        MPI_File_read(fhandle, num_rows, 1, MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);
        MPI_File_read(fhandle, num_cols, 1, MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);

        // Evaluate the number of rows associated to each rank and the offset in file
        unsigned int numRowsPerRank = *num_rows / nRanks;
        unsigned int offset = numRowsPerRank * sizeof(FloatingType) * myRank * *num_cols;
        if(myRank == nRanks - 1){
            numRowsPerRank += *num_rows % nRanks;
        }

        // File pointer is set to the current pointer position plus offset in order to read the right portion of the matrix
        MPI_File_seek(fhandle, offset, MPI_SEEK_CUR);

        // Evaluate the number of rows associated to each device in the rank
        unsigned int numRowsPerDevice = numRowsPerRank / numDevices;
        size_t s = 0;
        for(int i = 0; i < numDevices; i++){
            h_sizes[i] = (s + numRowsPerDevice <= numRowsPerRank) ? numRowsPerDevice : numRowsPerRank - s;
            s += numRowsPerDevice;
        }
        if(s < numRowsPerRank) h_sizes[numDevices - 1] += numRowsPerRank - s;

        // Allocate the space in each device for its chunk of the matrix
        #pragma omp parallel for num_threads(numDevices)
        for(int i = 0; i < numDevices; i++){
            cudaSetDevice(i);
            cudaMalloc(&A_dev[i], sizeof(FloatingType) * *num_cols * h_sizes[i]);
        }

        // Read matrix from file and copy it into the devices
        FloatingType ** h_A;
        cudaHostAlloc(&h_A, sizeof(FloatingType *) * numDevices, cudaHostAllocDefault);
        for(int k = 0; k < numDevices; k++) {
            cudaHostAlloc(&h_A[k], sizeof(FloatingType) * *num_cols * h_sizes[k], cudaHostAllocDefault);
            for (int i = 0; i < h_sizes[k]; i++) {
                for (int j = 0; j < *num_cols; j++){
                    MPI_File_read(fhandle, &h_A[k][i * (*num_cols) + j], 1, mpi_datatype, MPI_STATUS_IGNORE);
                }
            }
            cudaSetDevice(k);
            cudaMemcpyAsync(A_dev[k], h_A[k], sizeof(FloatingType) * *num_cols * h_sizes[k], cudaMemcpyHostToDevice, streams[k]);
        }

        // Close the file
        if(MPI_File_close(&fhandle) != MPI_SUCCESS) {
            printf("[MPI process %d] Failure in closing the file.\n", myRank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        printf("[MPI process %d] File closed successfully.\n", myRank);

        cudaFreeHost(h_A);
    }

}
#endif //CONJUGATEGRADIENT_MULTIGPUS_CUDA_NCCL_CUH