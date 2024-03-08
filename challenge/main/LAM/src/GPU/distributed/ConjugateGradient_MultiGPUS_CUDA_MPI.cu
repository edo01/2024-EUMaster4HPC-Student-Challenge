#include "ConjugateGradient_MultiGPUS_CUDA_MPI.cuh"

namespace LAM
{
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_BLOCKS=1000;
    constexpr int NUM_THREADS=1024;

    static uint64_t getHostHash(const char* string) {
    // Based on DJB2a, result = result * 33 ^ char
    uint64_t result = 5381;
    for (int c = 0; string[c] != '\0'; c++){
        result = ((result << 5) + result) ^ string[c];
    }
    return result;
    }


    static void getHostName(char* hostname, int maxlen) {
        gethostname(hostname, maxlen);
        for (int i=0; i< maxlen; i++) {
            if (hostname[i] == '.') {
                hostname[i] = '\0';
                return;
            }
        }
    }

    /**
     * ---------------------------------------------------------------
     * -----------------  CUDA Kernels  ------------------------------
     * ---------------------------------------------------------------
     */

    template<typename FloatingType>
    __global__ void divide(const FloatingType * a, const FloatingType * b, FloatingType * res)
    {
        *res = *a / *b;
    }

    template<typename FloatingType>
    __device__ void warpReduce(volatile FloatingType * a, int t)
    {
        unsigned int w = WARP_SIZE;
        while (w >= 1) {
            a[t] += a[t + w];
            w >>= 1;
        }
    }

    template<typename FloatingType>
    __global__ void reduce(const FloatingType * a, FloatingType * sum, size_t size)
    {
        // Allocate shared memory
        __shared__ FloatingType tmp[NUM_THREADS];
        // Retrieve global thread id
        unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
        // Load element
        if (tid < size){
            tmp[threadIdx.x] = a[tid];
        } else {
            tmp[threadIdx.x] = 0;
        }
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > WARP_SIZE; s /= 2){
            if(threadIdx.x < s){
                tmp[threadIdx.x] +=  tmp[threadIdx.x + s];
            }
            __syncthreads();
        }

        if(threadIdx.x < WARP_SIZE){
            warpReduce<FloatingType>(tmp, threadIdx.x);
        }

        // Let the thread 0 write its result to main memory
        if(threadIdx.x == 0){
            //!!sum non è un array
            sum[blockIdx.x] = tmp[0];
        }
    }

    template<typename FloatingType>
    __global__ void partialDot(const FloatingType * a, const FloatingType * b, FloatingType * partialDot, size_t size)
    {
        // Allocate shared memory
        __shared__ FloatingType tmp[NUM_THREADS];
        // Retrieve local and global thread id
        unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int t = threadIdx.x;
        // Load elements and check boundary condition
        if(tid < size){
            tmp[t] = a[tid] * b[tid];
        } else {
            tmp[t] = 0;
        }
        __syncthreads();
        // Reduce tmp
        for(unsigned int s = blockDim.x /2; s > WARP_SIZE; s /= 2){
            if(t < s){
                tmp[t] += tmp[t + s];
            }
            __syncthreads();
        }

        if(t < WARP_SIZE){
            warpReduce<FloatingType>(tmp, t);
        }

        // Let the thread 0 for this block write its result to main memory
        if(t == 0) {
            partialDot[blockIdx.x] = tmp[0];
        }
    }

    template<typename FloatingType>
    __host__ void dot(const FloatingType* a, const FloatingType* b, FloatingType* res, size_t size, cudaStream_t stream)
    {
        FloatingType * partialSum;
        //!! I think this should be changed with cudaMallocAsync
        cudaMalloc(&partialSum, sizeof(FloatingType) * NUM_BLOCKS);
        partialDot<FloatingType><<<NUM_BLOCKS,NUM_THREADS, 0, stream>>>(a, b, partialSum, size);
        //!! qui dovrebbe essere NUM_BLOCKS e non NUM_THREADS perché partialSum è un array di NUM_BLOCKS elementi
        reduce<FloatingType><<<1,NUM_THREADS, 0, stream>>>(partialSum, res, NUM_BLOCKS); // TODO deal with the case in which only one reduce is not enough
        
        cudaFree(partialSum);
    }

    template<typename FloatingType>
    __global__ void axpby(const FloatingType * alpha, const FloatingType * x, const FloatingType * beta, FloatingType * y, size_t size)
    {
        // y = alpha * x + beta * y
        // Retrieve global thread id
        unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int stride = blockDim.x * gridDim.x;
        // Check boundary
        for(unsigned int i = tid; i < size; i += stride){
            y[tid] *= *beta;
            y[tid] += *alpha * x[tid];
        }
    }

    template<typename FloatingType>
    __global__ void axpy(const FloatingType * alpha, const FloatingType * x, FloatingType * y, size_t size)
    {
        // y = alpha * x +  y
        // Retrieve global thread id
        unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int stride = blockDim.x * gridDim.x;
        // Check boundary
        for(unsigned int i = tid; i < size; i += stride){
            y[tid] += *alpha * x[tid];
        }
    }

    template<typename FloatingType>
    __global__ void minusaxpy(const FloatingType * alpha, const FloatingType * x, FloatingType * y, size_t size)
    {
        // y = alpha * x +  y
        // Retrieve global thread id
        unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int stride = blockDim.x * gridDim.x;
        // Check boundary
        for(unsigned int i = tid; i < size; i += stride) {
            y[tid] -= *alpha * x[tid];
        }
    }

    template<typename FloatingType>
    __global__ void xpby( const FloatingType * x, const FloatingType *beta, FloatingType * y, size_t size)
    {
        // y = alpha * x + beta * y
        // Retrieve global thread id
        unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned int stride = blockDim.x * gridDim.x;
        // Check boundary
        for(unsigned int i = tid; i < size; i += stride){
            y[tid] *= *beta;
            y[tid] += x[tid];
        }
    }

    template<typename FloatingType>
    __global__ void gemv(FloatingType alpha, const FloatingType * A, const FloatingType * x, FloatingType beta, FloatingType * y, size_t num_rows, size_t num_cols, unsigned int stride_row)
    {
        // y = alpha * A * x + beta * y

        // Allocate shared memory
        __shared__ FloatingType tmp[NUM_THREADS];
        // Retrieve ids
        unsigned int row = blockIdx.x + stride_row;
        unsigned int col = threadIdx.x;
        // Number of elements each thread has to load
        unsigned int n = (num_cols % NUM_THREADS == 0) ? num_cols / NUM_THREADS : num_cols / NUM_THREADS + 1;
        // Load elements
        tmp[col] = 0;
        while(n > 0){
            size_t col_new = col + (n - 1) * blockDim.x;

            if(col_new < num_cols && row < num_rows) {
                tmp[col] += alpha * A[row * num_cols + col_new] * x[col_new];
            }
            n--;
        }
        __syncthreads();

        // Reduce tmp
        for (unsigned int s = blockDim.x / 2; s > WARP_SIZE; s /= 2){
            if (col < s){
                tmp[col] += tmp[col + s];
            }
            __syncthreads();
        }

        if(col < WARP_SIZE){
            warpReduce<FloatingType>(tmp, col);
        }

        // Let the thread 0 within the block write the partial reduction into main memory
        if(col == 0 && row < num_rows) {
            y[row] *= beta;
            y[row] += tmp[0];
        }
    }

    template<typename FloatingType>
    __host__ void gemv_host(FloatingType alpha, const FloatingType * A, FloatingType * x, FloatingType beta, FloatingType * y, size_t num_rows, size_t num_cols, cudaStream_t stream)
    {
        unsigned int processedRow = 0;
        unsigned int numKernels = (num_rows % NUM_BLOCKS == 0) ? num_rows / NUM_BLOCKS : num_rows / NUM_BLOCKS + 1;
        while(numKernels > 0){
            gemv<FloatingType><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(alpha, A, x, beta, y, num_rows, num_cols, processedRow);
            processedRow += NUM_BLOCKS;
            numKernels--;
        }
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_MPI<FloatingType>::solve(int max_iters, FloatingType rel_error){
        //initializing MPI
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        // since only the matrix-vector multiplication is parallelized,
        // these variables are only needed in the device 0 of rank 0
        FloatingType * b_dev, * x_dev, * alpha_dev, * beta_dev, * bb_dev;
        FloatingType * rr_dev, * rr_new_dev, * r_dev, * pAp_dev, * Ap0_dev;
        FloatingType * rr, * bb;

        bool stop = false;
        rr = new FloatingType;
        bb = new FloatingType;
        int num_iters=1;

        /*
        *  ---------------------------------------------------------------
        *  -----------------  Device memory allocation  ------------------
        *  ---------------------------------------------------------------
        */
        /* Allocate and initialize a given number of rows of A to each device
           data for computating the matrix-vector multiplication */
        
        // stores the pointer to the result of the matrix-vector multiplication of the i-th device
        FloatingType * Ap_dev;

        /* stores the pointer to the vector for the matrix-vector multiplication of the i-th device
           of the i-th device */
        FloatingType * p_dev;

        cudaSetDevice(_device_id);

        cudaMalloc(&Ap_dev, _num_local_rows * sizeof(FloatingType));
        cudaMalloc(&p_dev, _num_cols * sizeof(FloatingType));


        // Allocate ONLY in device 0 of rank 0 the other variables
        if (rank == 0) {
            cudaMalloc(&b_dev, sizeof(FloatingType) * _num_cols);
            cudaMalloc(&x_dev, sizeof(FloatingType) * _num_cols);
            cudaMalloc(&alpha_dev, sizeof(FloatingType));
            cudaMalloc(&beta_dev, sizeof(FloatingType));
            cudaMalloc(&bb_dev, sizeof(FloatingType));
            cudaMalloc(&rr_dev, sizeof(FloatingType));
            cudaMalloc(&rr_new_dev, sizeof(FloatingType));
            cudaMalloc(&r_dev, sizeof(FloatingType) * _num_cols);
            cudaMalloc(&pAp_dev, sizeof(FloatingType));
            cudaMalloc(&Ap0_dev, sizeof(FloatingType) * _num_cols); // Ap0_dev is located in device 0 and will collect all the result from the devices

            cudaMemcpyAsync(b_dev, _rhs, sizeof(FloatingType) * _num_cols, cudaMemcpyHostToDevice, stream);
            cudaMemsetAsync(x_dev, 0, sizeof(FloatingType) * _num_cols, stream); // x = 0
            cudaMemcpyAsync(r_dev, b_dev, sizeof(FloatingType) * _num_cols, cudaMemcpyDeviceToDevice, stream); // r = b
            cudaMemcpyAsync(p_dev, b_dev, sizeof(FloatingType) * _num_cols, cudaMemcpyDeviceToDevice, stream); // p = b

            dot<FloatingType>(b_dev, b_dev, bb_dev, _num_cols, stream); // bb = b * b
            
            //print bb_dev
            cudaMemcpy(bb, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);

            cudaMemcpyAsync(rr_dev, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToDevice, stream); // rr = bb
        }


        /*
        *  ---------------------------------------------------------------
        *  -----------------  CG Algorithm  ------------------------------
        *  ---------------------------------------------------------------
        */ 
        // calc the average time of each iteration
        std::chrono::duration<double> avg_time_gemv(0);
        std::chrono::duration<double> avg_time_cg(0);

        // CG Iterations
        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {
            auto cg_start = std::chrono::high_resolution_clock::now();

            auto gemv_start = std::chrono::high_resolution_clock::now();
            // MPI broadcast of p_dev to all other ranks
            MPI_Bcast(p_dev, _num_cols, get_mpi_datatype(), 0, MPI_COMM_WORLD);

            // Performs matrix-vector multiplication in each device of each rank
            gemv_host<FloatingType>(1.0, _A_dev, p_dev, 0.0, Ap_dev, _num_local_rows, _num_cols, stream);

            cudaDeviceSynchronize();

            MPI_Gatherv(Ap_dev, _num_local_rows, get_mpi_datatype(), Ap0_dev, _sendcounts, _displs, get_mpi_datatype(), 0, MPI_COMM_WORLD);
            
            auto gemv_end = std::chrono::high_resolution_clock::now();
            avg_time_gemv += gemv_end - gemv_start;

            // Device 0 in rank 0 carries on all the other operation involved in the iteration of the CG method
            if(rank == 0) {

                dot<FloatingType>(p_dev, Ap0_dev, pAp_dev, _num_cols, stream);

                divide<FloatingType><<<1, 1, 0, stream>>>(rr_dev, pAp_dev, alpha_dev);

                axpy<FloatingType><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>
                        (alpha_dev, p_dev, x_dev, _num_cols);

                minusaxpy<FloatingType><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>
                        (alpha_dev, Ap0_dev, r_dev, _num_cols);

                
                dot<FloatingType>(r_dev, r_dev, rr_new_dev, _num_cols, stream);

               
                divide<FloatingType><<<1, 1, 0, stream>>>(rr_new_dev, rr_dev, beta_dev);
                
                cudaMemcpy(rr_dev, rr_new_dev, sizeof(FloatingType), cudaMemcpyDeviceToDevice);
                cudaMemcpy(rr, rr_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);
                cudaMemcpy(bb, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);


                cudaDeviceSynchronize();

                if (std::sqrt(*rr / *bb) < rel_error) { stop = true; }

            }

            // Rank 0 broadcasts the flag stop to all other rank in order to stop the computation when the solution is found
            MPI_Bcast(&stop, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
            if(stop) { 
                break;
            }

            // Device 0 in rank 0 computes the new value of p that will be broadcast to all other devices in the next iteration
            if (rank == 0){
                xpby<FloatingType><<<NUM_BLOCKS, NUM_THREADS, 0, stream>>>(r_dev, beta_dev, p_dev, _num_cols);
            }

            auto cg_end = std::chrono::high_resolution_clock::now();
            avg_time_cg += cg_end - cg_start;
        }

        //print the average time of each iteration
        //PRINT_RANK0("gemv:%f,", avg_time_gemv.count() / num_iters);
        //PRINT_RANK0("cg:%f,", avg_time_cg.count() / num_iters);
        if(rank == 0) {
            std::cout << avg_time_gemv.count() / num_iters << ",";
            std::cout << avg_time_cg.count() / num_iters << ",";
        }

        /*
         * ---------------------------------------------------------------
         * ------------- Freeing memory and save the solution ------------
         * ---------------------------------------------------------------
         */

        // Device 0 of rank 0 prints the information about the result of the CG method
        if(rank == 0) {

            // Prints the number of iterations and the relative error
            /*if (num_iters <= max_iters) {
                printf("PARALLEL MULTI-GPUS CUDA MPI: Converged in %d iterations, relative error is %e\n", num_iters,
                       std::sqrt(*rr / *bb));
            } else {
                printf("PARALLEL MULTI-GPUS CUDA MPI: Did not converge in %d iterations, relative error is %e\n", max_iters,
                       std::sqrt(*rr / *bb));
            }*/

            if(rank == 0) {
                std::cout << num_iters << ",";
                std::cout << std::sqrt(*rr / *bb) << ",";
            }

            // Copy solution to host
            cudaMemcpyAsync(_x, x_dev, _num_cols * sizeof(FloatingType), cudaMemcpyDeviceToHost, stream);

            // Free GPU memory
            cudaFreeAsync(alpha_dev, stream);
            cudaFreeAsync(beta_dev, stream);
            cudaFreeAsync(bb_dev, stream);
            cudaFreeAsync(rr_dev, stream);
            cudaFreeAsync(rr_new_dev, stream);
            cudaFreeAsync(r_dev, stream);
            cudaFreeAsync(p_dev, stream);
            cudaFreeAsync(Ap0_dev, stream);
            cudaFreeAsync(pAp_dev, stream);
        }

        // Free CPU memory
        delete bb;
        delete rr;

        cudaFreeAsync(Ap_dev, stream);
        cudaFreeAsync(p_dev, stream);

        return (num_iters <= max_iters);
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_MPI<FloatingType>::load_matrix_from_file(const char * filename){
        int rank, num_procs, localRank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        /*
        * ---------------------------------------------------------------
        * -----------------  Retrieve the device id  -------------------
        * --------------------------------------------------------------
        */

        uint64_t hostHashs[num_procs];
        char hostname[1024];
        getHostName(hostname, 1024);
        hostHashs[rank] = getHostHash(hostname);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
        for (int p=0; p<num_procs; p++) {
            if (p == rank) break;
            if (hostHashs[p] == hostHashs[rank]) localRank++;
        }

        // from now on I am going to work only with the device localRank
        _device_id = localRank;
        cudaSetDevice(_device_id);

        // create a stream for the device
        cudaStreamCreate(&stream);

        MPI_File fhandle;
        size_t num_total_rows;
        size_t file_offset;
        //size_t numRowsPerDevice;
        
        // it is just a temporary variable to store the portion of the matrix that will be transfered
        FloatingType* h_A;

        // Initialize an MPI file handler and open the file
        if(MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandle) != MPI_SUCCESS) {
            printf("[MPI process %d] Failure in opening the file.\n", rank);
            //MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return false;
        }
        //printf("[MPI process %d] File opened successfully.\n", rank);

        // Read from file the dimensions of the matrix
        MPI_File_read(fhandle, &num_total_rows, 1, MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);
        MPI_File_read(fhandle, &_num_cols, 1, MPI_UNSIGNED_LONG, MPI_STATUS_IGNORE);

        if(rank==0){
            std::cout << num_total_rows << ",";
        }

        // Evaluate the number of rows associated to each rank and the offset in file
        _num_local_rows = num_total_rows / num_procs;
        file_offset = _num_local_rows * sizeof(FloatingType) * rank * _num_cols;

        //the last rank will have the remaining rows
        if(rank == num_procs - 1){
            //add the reminder to the last rank
            _num_local_rows += num_total_rows % num_procs; 
        }

        _sendcounts = new int[num_procs]; 
        _displs = new int[num_procs];

        for(int i=0; i<num_procs; i++){
            _sendcounts[i] = num_total_rows/num_procs;
            _sendcounts[i] += (i==num_procs-1) ? num_total_rows%num_procs : 0;
            _displs[i] = i * (num_total_rows/num_procs);
        }

        // File pointer is set to the current pointer position plus offset in order to read the right portion of the matrix
        MPI_File_seek(fhandle, file_offset, MPI_SEEK_CUR);
        
        //size_t DATA = _num_cols * _num_local_rows * sizeof(FloatingType);
        //PRINT_RANK0("I am trying to allocate %lu bytes (%f GB)\n", DATA, DATA / (1024.0 * 1024.0 * 1024.0));

        cudaMalloc(&_A_dev, sizeof(FloatingType) * _num_cols * _num_local_rows);

        // allocate the space in the host (using pinned memory) and read the portion of the matrix from the file
        cudaHostAlloc(&h_A, sizeof(FloatingType) * _num_cols * _num_local_rows, cudaHostAllocDefault);
        MPI_File_read(fhandle, &h_A[0], _num_cols * _num_local_rows, get_mpi_datatype(), MPI_STATUS_IGNORE);
        // copy the portion of the matrix into the device
        cudaMemcpyAsync(_A_dev, h_A, sizeof(FloatingType) * _num_cols * _num_local_rows, cudaMemcpyHostToDevice, stream);

        // Close the file
        if(MPI_File_close(&fhandle) != MPI_SUCCESS) {
            printf("[MPI process %d] Failure in closing the file.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return false;
        }

        cudaFreeHost(h_A);
        return true;
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_MPI<FloatingType>::load_rhs_from_file(const char * filename){
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        if(rank!=0) return true;  // Only rank 0 reads the right-hand side vector from the matrix

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
        if(rhs_rows != _num_cols)
        {
            fprintf(stderr, "Size of right hand side does not match the matrix\n");
            return false;
        }

        cudaHostAlloc(&_rhs, sizeof(FloatingType) * rhs_rows, cudaHostAllocDefault);
        cudaHostAlloc(&_x, sizeof(FloatingType) * rhs_rows, cudaHostAllocDefault);
        

        fread(_rhs, sizeof(FloatingType), rhs_rows, file);

        fclose(file);

        return true;
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_MPI<FloatingType>::generate_matrix(const size_t num_rows, const size_t num_cols){
        int rank, num_procs, localRank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        /*
        * ---------------------------------------------------------------
        * -----------------  Retrieve the device id  -------------------
        * --------------------------------------------------------------
        */

        uint64_t hostHashs[num_procs];
        char hostname[1024];
        getHostName(hostname, 1024);
        hostHashs[rank] = getHostHash(hostname);
        MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
        for (int p=0; p<num_procs; p++) {
            if (p == rank) break;
            if (hostHashs[p] == hostHashs[rank]) localRank++;
        }

        // from now on I am going to work only with the device localRank
        _device_id = localRank;
        cudaSetDevice(_device_id);

        // create a stream for the device
        cudaStreamCreate(&stream);

        size_t num_total_rows = num_rows;
        _num_cols = num_rows;
        size_t offset;
        //size_t numRowsPerDevice;

        //PRINT_RANK0("%lu,", num_total_rows);
        if(rank==0){
            std::cout << num_total_rows << ",";
        }

        // it is just a temporary variable to store the portion of the matrix that will be transfered
        FloatingType* h_A;

        // Evaluate the number of rows associated to each rank and the offset in file
        _num_local_rows = num_total_rows / num_procs;
        offset = _num_local_rows * rank;    

        //the last rank will have the remaining rows
        if(rank == num_procs - 1){
            //add the reminder to the last rank
            _num_local_rows += num_total_rows % num_procs; 
        }

        _sendcounts = new int[num_procs]; 
        _displs = new int[num_procs];

        for(int i=0; i<num_procs; i++){
            _sendcounts[i] = num_total_rows/num_procs;
            _sendcounts[i] += (i==num_procs-1) ? num_total_rows%num_procs : 0;
            _displs[i] = i * (num_total_rows/num_procs);
        }
        
        //long unsigned int DATA_SIZE = _num_local_rows * _num_cols * sizeof(FloatingType);
        //PRINT_RANK0("Problem size: %lu bytes (%f GB)\n", DATA_SIZE, DATA_SIZE/(1024.0*1024.0*1024.0));
        //DATA_SIZE += _num_cols*5*sizeof(FloatingType);
        //PRINT_RANK0("I am trying to allocate %lu bytes (%f GB)\n", DATA_SIZE, DATA_SIZE/(1024.0*1024.0*1024.0));
        fflush(stdout);

        cudaMalloc(&_A_dev, sizeof(FloatingType) * _num_cols * _num_local_rows);

        // allocate the space in the host (using pinned memory) and read the portion of the matrix from the file
        cudaHostAlloc(&h_A, sizeof(FloatingType) * _num_cols * _num_local_rows, cudaHostAllocDefault);
        
        for (size_t i = 0; i < _num_local_rows; i++) {
            for (size_t j = 0; j < _num_cols; j++) {
                if(i+offset==j-1 || i+offset==j+1) 
                    h_A[i * _num_cols + j] = 1;
                else if(i+offset==j)
                    h_A[i * _num_cols + j] = 2;
                else
                    h_A[i * _num_cols + j] = 0;
            }
        }
        
        // copy the portion of the matrix into the device
        cudaMemcpyAsync(_A_dev, h_A, sizeof(FloatingType) * _num_cols * _num_local_rows, cudaMemcpyHostToDevice, stream);

        cudaFreeHost(h_A);
        return true;
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_MPI<FloatingType>::generate_rhs(){
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        size_t rhs_rows;

        rhs_rows = _num_cols;

        cudaHostAlloc(&_rhs, sizeof(FloatingType) * rhs_rows, cudaHostAllocDefault);
        cudaHostAlloc(&_x, sizeof(FloatingType) * rhs_rows, cudaHostAllocDefault);
        
        for (int i = 0; i < rhs_rows; i++)
        {
            _rhs[i] = 1.0;
        }

        return true;
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_MPI<FloatingType>::save_result_to_file(const char * filename) const
    {
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
     
        if(rank!=0) return true;  // Only rank 0 writes the result to the file
     
        FILE * file = fopen(filename, "wb");
        if(file == nullptr)
        {
            fprintf(stderr, "Cannot open output file\n");
            return false;
        }
        int num_rows = _num_cols;
        int num_cols = 1;
        fwrite(&num_rows, sizeof(size_t), 1, file);
        fwrite(&num_cols, sizeof(size_t), 1, file);
        //save rhs to file
        fwrite(_x, sizeof(FloatingType), num_rows, file);

        fclose(file);

        return true;
    }

template class LAM::ConjugateGradient_MultiGPUS_CUDA_MPI<float>;
template class LAM::ConjugateGradient_MultiGPUS_CUDA_MPI<double>;

}