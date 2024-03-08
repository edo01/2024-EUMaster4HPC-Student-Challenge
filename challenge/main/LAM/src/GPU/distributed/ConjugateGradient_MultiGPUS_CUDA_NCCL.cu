#include "ConjugateGradient_MultiGPUS_CUDA_NCCL.cuh"

namespace LAM
{
    constexpr int WARP_SIZE = 32;

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
        tmp[t] = 0;
        while(tid < size) {
            tmp[t] += a[tid] * b[tid];
            tid += blockDim.x * gridDim.x;
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
        cudaMalloc(&partialSum, sizeof(FloatingType) * NUM_BLOCKS);
        partialDot<FloatingType><<<NUM_BLOCKS,NUM_THREADS, 0, stream>>>(a, b, partialSum, size);
        int s = gridSize;
        while( s > 1 ){
            if( s < blockSize ){
                reduce<FloatingType, gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(partialSum, res, s);
            } else {
                reduce<FloatingType, gridSize, blockSize><<<gridSize, blockSize, 0, stream>>>(partialSum, partialSum, s);
            }
            s = (s % blockSize == 0 && s != 2) ? s/blockSize : s/blockSize + 1;
        }
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
    bool ConjugateGradient_MultiGPUS_CUDA_NCCL<FloatingType>::solve(int max_iters, FloatingType rel_error)
    {
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
        FloatingType * Ap_dev[_numDevices];

        /* stores the pointer to the vector for the matrix-vector multiplication of the i-th device
           of the i-th device */
        FloatingType * p_dev[_numDevices];

        // Allocate p and Ap on all devices
        #pragma omp parallel for num_threads(_numDevices)
        for(int i = 0; i < _numDevices; i++){
            cudaSetDevice(i);
 
            cudaMalloc(&Ap_dev[i], _rows_per_device[i] * sizeof(FloatingType));
            cudaMalloc(&p_dev[i], _num_cols * sizeof(FloatingType));
        }

        // Allocate the in device 0 of rank 0 the other variables
        if (rank == 0) {
            cudaSetDevice(0);
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

            cudaMemcpyAsync(b_dev, _rhs, sizeof(FloatingType) * _num_cols, cudaMemcpyHostToDevice, streams[0]);
            cudaMemsetAsync(x_dev, 0, sizeof(FloatingType) * _num_cols, streams[0]); // x = 0
            cudaMemcpyAsync(r_dev, b_dev, sizeof(FloatingType) * _num_cols, cudaMemcpyDeviceToDevice, streams[0]); // r = b
            cudaMemcpyAsync(p_dev[0], b_dev, sizeof(FloatingType) * _num_cols, cudaMemcpyDeviceToDevice, streams[0]); // p = b

            dot<FloatingType>(b_dev, b_dev, bb_dev, _num_cols, streams[0]); // bb = b * b
            cudaMemcpyAsync(rr_dev, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToDevice, streams[0]); // rr = bb
        }

        /*
        *  ---------------------------------------------------------------
        *  -----------------  NCCL initialization  -----------------------
        *  ---------------------------------------------------------------
        */
        int tot_num_devices = _numDevices * num_procs;
        
        // id is used to identify a NCCL communication group
        ncclUniqueId id;
        
        /*
         * Each ncclComm_t is associated with a specific GPU device because NCCL
         * is designed to perform collective communication operations among a group 
         * of GPUs. Each GPU in the group needs to have its own unique communicator 
         * to manage its part of the collective operation.
        */
        ncclComm_t comms[_numDevices];
        
        // Generating NCCL unique ID at one process and broadcasting it to all
        if (rank == 0) ncclGetUniqueId(&id);
        MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        //  Initializing NCCL
        ncclGroupStart();
        for (int i=0; i<_numDevices; i++) {
            cudaSetDevice(i);
            int device_rank = rank * _numDevices + i;

            // ncclCommInitRank is used to create a new communicator for each device
            // it takes the communicator, the number of devices in the NCCL group,
            // the unique ID of the NCCL communication group, the rank of
            // the devices in the NCCL group
            ncclCommInitRank(&comms[i], tot_num_devices, id, device_rank);
        }
        ncclGroupEnd();

        /*
        *  ---------------------------------------------------------------
        *  -----------------  CG Algorithm  ------------------------------
        *  ---------------------------------------------------------------
        */ 

        // CG Iterations
        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {

            // Copy p from device 0 to all other devices to allow matrix-vector multiplication
            ncclGroupStart();
            for (int i=0; i < _numDevices; i++) {
                ncclBroadcast(p_dev[0], p_dev[i], _num_cols, nccl_datatype, 0, comms[i], streams[i]);
            }
            ncclGroupEnd();

            // Performs matrix-vector multiplication in each device of each rank
            #pragma omp parallel for num_threads(_numDevices)
            for(int i = 0; i < _numDevices; i++){
                cudaSetDevice(i);
                gemv_host<FloatingType>(1.0, _A_dev[i], p_dev[i], 0.0, Ap_dev[i], _rows_per_device[i], _num_cols, streams[i]);
            }

            // All-To-One Gather to collect all the results of the mat-vec multiplication in device 0 in rank 0
            ncclGroupStart();
            if(rank == 0) {
                int offset = 0;
                // for each device, the rank 0 collects the result of the mat-vec multiplication with a recv
                for(int i = 0; i < _numDevices * num_procs; i++){
                    if(i < _numDevices * (num_procs - 1)) {
                        // devices of rank 0 to num_procs - 2 have the same number of rows and so the same distribution among devices
                        ncclRecv(Ap0_dev + offset, _rows_per_device[i % _numDevices], nccl_datatype, i, comms[0], streams[0]);
                        offset += _rows_per_device[i % _numDevices];
                    } else {
                        // the last rank can have a different number of rows
                        unsigned int numRowsLastRank = _num_cols / num_procs + _num_cols % num_procs;
                        unsigned int numRowsDeviceLastRank = numRowsLastRank / _numDevices;
                        if (i == _numDevices * num_procs - 1){
                            // the last device of the last rank has the remaining rows
                            numRowsDeviceLastRank += numRowsLastRank % _numDevices;
                        }
                        ncclRecv(Ap0_dev + offset, numRowsDeviceLastRank, nccl_datatype, i, comms[0], streams[0]);
                        offset += numRowsDeviceLastRank;
                    }
                }
            }
            for(int i = 0; i < _numDevices; i++) {
                ncclSend(Ap_dev[i], _rows_per_device[i] , nccl_datatype, 0, comms[i], streams[i]);
            }
            ncclGroupEnd();

            // Synchronizing on CUDA stream to complete NCCL communication
            for (int i = 0; i < _numDevices; i++) {
                cudaStreamSynchronize(streams[i]);
            }

            // Device 0 in rank 0 carries on all the other operation involved in the iteration of the CG method
            if(rank == 0) {
                cudaSetDevice(0);

                dot<FloatingType>(p_dev[0], Ap0_dev, pAp_dev, _num_cols, streams[0]);

                divide<FloatingType><<<1, 1, 0, streams[0]>>>(rr_dev, pAp_dev, alpha_dev);

                axpy<FloatingType><<<NUM_BLOCKS, NUM_THREADS, 0, streams[0]>>>
                        (alpha_dev, p_dev[0], x_dev, _num_cols);

                minusaxpy<FloatingType><<<NUM_BLOCKS, NUM_THREADS, 0, streams[0]>>>
                        (alpha_dev, Ap0_dev, r_dev, _num_cols);

                
                dot<FloatingType>(r_dev, r_dev, rr_new_dev, _num_cols, streams[0]);

               
                divide<FloatingType><<<1, 1, 0, streams[0]>>>(rr_new_dev, rr_dev, beta_dev);

                
                //cudaMemcpyAsync(rr_dev, rr_new_dev, sizeof(FloatingType), cudaMemcpyDeviceToDevice, streams[0]);

                //cudaMemcpyAsync(rr, rr_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost, streams[0]);
                //cudaMemcpyAsync(bb, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost, streams[0]);

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
                cudaSetDevice(0);
                xpby<FloatingType><<<NUM_BLOCKS, NUM_THREADS, 0, streams[0]>>>(r_dev, beta_dev, p_dev[0], _num_cols);
            }
        }

        /*
         * ---------------------------------------------------------------
         * ------------- Freeing memory and save the solution ------------
         * ---------------------------------------------------------------
         */

        // Device 0 of rank 0 prints the information about the result of the CG method
        if(rank == 0) {
            //cudaSetDevice(0);
            //cudaMemcpy(rr, rr_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);
            //cudaMemcpy(bb, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);

            // Prints the number of iterations and the relative error
            if (num_iters <= max_iters) {
                printf("PARALLEL MULTI-GPUS CUDA NCCL: Converged in %d iterations, relative error is %e\n", num_iters,
                       std::sqrt(*rr / *bb));
            } else {
                printf("PARALLEL MULTI-GPUS CUDA NCCL: Did not converge in %d iterations, relative error is %e\n", max_iters,
                       std::sqrt(*rr / *bb));
            }

            // Copy solution to host
            cudaSetDevice(0);
            cudaMemcpyAsync(_x, x_dev, _num_cols * sizeof(FloatingType), cudaMemcpyDeviceToHost, streams[0]);

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
        }

        // All devices free their allocated memory and destroy streams
        #pragma omp parallel for num_threads(_numDevices)
        for(int i = 0; i < _numDevices; i++){
            cudaSetDevice(i);
            cudaFreeAsync(Ap_dev[i], streams[i]);
            cudaFreeAsync(p_dev[i], streams[i]);
        }

        // Finalizing NCCL
        for (int i = 0; i < _numDevices; i++) {
            ncclCommDestroy(comms[i]);
        }

        return (num_iters <= max_iters);
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_NCCL<FloatingType>::load_matrix_from_file(const char * filename)
    {
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        MPI_File fhandle;
        size_t num_total_rows;
        size_t file_offset;
        size_t numRowsPerDevice;
        

        // at the index i, it contains a portion of the matrix that will be transfered to the i-th device
        // it is just a temporary variable to store the portion of the matrix that will be transfered
        FloatingType** h_A;

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

        // Evaluate the number of rows associated to each rank and the offset in file
        _num_local_rows = num_total_rows / num_procs;
        file_offset = _num_local_rows * sizeof(FloatingType) * rank * _num_cols;
        _offset = _num_local_rows * rank;

        //the last rank will have the remaining rows
        if(rank == num_procs - 1){
            //add the reminder to the last rank
            _num_local_rows += num_total_rows % num_procs; 
        }

        // File pointer is set to the current pointer position plus offset in order to read the right portion of the matrix
        MPI_File_seek(fhandle, file_offset, MPI_SEEK_CUR);

        // Allocates page-locked memory on the host for asynchronous memory copy
        cudaHostAlloc(&_rows_per_device, sizeof(size_t) * _numDevices, cudaHostAllocDefault);


        // Evaluate the number of rows associated to each device in the rank
        numRowsPerDevice = _num_local_rows / _numDevices;
        size_t s = 0;
        for(int i = 0; i < _numDevices; i++){
            // The last device will have the remaining rows
            _rows_per_device[i] = (s + numRowsPerDevice <= _num_local_rows) ? numRowsPerDevice : _num_local_rows - s;
            s += numRowsPerDevice;
        }
        if(s < _num_local_rows) _rows_per_device[_numDevices - 1] += _num_local_rows - s;
        
        cudaHostAlloc(_A_dev, sizeof(FloatingType *) * _numDevices, cudaHostAllocDefault);

        // Allocate the space in each device for its chunk of the matrix
        #pragma omp parallel for num_threads(_numDevices)
        for(int i = 0; i < _numDevices; i++){
            cudaSetDevice(i);
            cudaMalloc(&_A_dev[i], sizeof(FloatingType) * _num_cols * _rows_per_device[i]);
        }

        // Read matrix from file and copy it into the devices
        cudaHostAlloc(&h_A, sizeof(FloatingType *) * _numDevices, cudaHostAllocDefault);
        /* for each device, allocate the space in the host and read the portion of the matrix from the file
           then copy it into the device*/ 
        for(int k = 0; k < _numDevices; k++) {
            cudaHostAlloc(&h_A[k], sizeof(FloatingType) * _num_cols * _rows_per_device[k], cudaHostAllocDefault);
            MPI_File_read(fhandle, &h_A[k][0], _num_cols*_rows_per_device[k], get_mpi_datatype(), MPI_STATUS_IGNORE);
            cudaSetDevice(k);
            cudaMemcpyAsync(_A_dev[k], h_A[k], sizeof(FloatingType) * _num_cols * _rows_per_device[k], cudaMemcpyHostToDevice, streams[k]);
        }

        // Close the file
        if(MPI_File_close(&fhandle) != MPI_SUCCESS) {
            printf("[MPI process %d] Failure in closing the file.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            return false;
        }
        //printf("[MPI process %d] File closed successfully.\n", rank);

        for(int i = 0; i < _numDevices; i++){
            cudaSetDevice(i);
            cudaFreeHost(h_A[i]);
        }

        cudaFreeHost(h_A);
        return true;
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_NCCL<FloatingType>::load_rhs_from_file(const char * filename)
    {
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
    bool ConjugateGradient_MultiGPUS_CUDA_NCCL<FloatingType>::generate_matrix(size_t num_rows, size_t num_cols)
    {
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        size_t num_total_rows;
        //size_t file_offset;
        size_t numRowsPerDevice;

        // at the index i, it contains a portion of the matrix that will be transfered to the i-th device
        // it is just a temporary variable to store the portion of the matrix that will be transfered
        FloatingType** h_A;

        num_total_rows = num_rows;
        _num_cols = num_rows;

        // Evaluate the number of rows associated to each rank and the offset in file
        _num_local_rows = num_total_rows / num_procs;
        _offset = _num_local_rows * rank;

        //the last rank will have the remaining rows
        if(rank == num_procs - 1){
            //add the reminder to the last rank
            _num_local_rows += num_total_rows % num_procs; 
        }

        // Allocates page-locked memory on the host for asynchronous memory copy
        cudaHostAlloc(&_rows_per_device, sizeof(size_t) * _numDevices, cudaHostAllocDefault);


        // Evaluate the number of rows associated to each device in the rank
        numRowsPerDevice = _num_local_rows / _numDevices;
        size_t s = 0;
        for(int i = 0; i < _numDevices; i++){
            // The last device will have the remaining rows
            _rows_per_device[i] = (s + numRowsPerDevice <= _num_local_rows) ? numRowsPerDevice : _num_local_rows - s;
            s += numRowsPerDevice;
        }
        if(s < _num_local_rows) _rows_per_device[_numDevices - 1] += _num_local_rows - s;

        cudaHostAlloc(_A_dev, sizeof(FloatingType *) * _numDevices, cudaHostAllocDefault);
        // Allocate the space in each device for its chunk of the matrix
        //#pragma omp parallel for num_threads(_numDevices)
        for(int i = 0; i < _numDevices; i++){
            cudaSetDevice(i);
            cudaMalloc(&_A_dev[i], sizeof(FloatingType) * _num_cols * _rows_per_device[i]);
        }
        // Read matrix from file and copy it into the devices
        cudaHostAlloc(&h_A, sizeof(FloatingType *) * _numDevices, cudaHostAllocDefault);
        /* for each device, allocate the space in the host and read the portion of the matrix from the file
           then copy it into the device*/ 
        for(int k = 0; k < _numDevices; k++) {
            cudaHostAlloc(&h_A[k], sizeof(FloatingType) * _num_cols * _rows_per_device[k], cudaHostAllocDefault);
            //MPI_File_read(fhandle, &h_A[k][0], _num_cols*_rows_per_device[k], get_mpi_datatype(), MPI_STATUS_IGNORE);
            for (size_t i = 0; i < _rows_per_device[k]; i++) {
                for (size_t j = 0; j < _num_cols; j++) {
                    if((i+_offset+numRowsPerDevice*k)==j-1 || (i+_offset+numRowsPerDevice*k)==j+1) 
                        h_A[k][i * _num_cols + j] = 1;
                    else if((i+_offset+numRowsPerDevice*k)==j)
                        h_A[k][i * _num_cols + j] = 2;
                    else
                        h_A[k][i * _num_cols + j] = 0;
                }
            }
            
            cudaSetDevice(k);
            // I'm pretty sure that these copies are done in serial, so it is not efficient 
            cudaMemcpyAsync(_A_dev[k], h_A[k], sizeof(FloatingType) * _num_cols * _rows_per_device[k], cudaMemcpyHostToDevice, streams[k]);
        }

        //printf("[MPI process %d] File closed successfully.\n", rank);

        for(int i = 0; i < _numDevices; i++){
            cudaSetDevice(i);
            cudaFreeHost(h_A[i]);
        }

        cudaFreeHost(h_A);
        return true;
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_NCCL<FloatingType>::generate_rhs()
    {
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        if(rank!=0) return true;  // Only rank 0 reads the right-hand side vector from the matrix

    
        size_t rhs_rows = _num_cols;

        cudaHostAlloc(&_rhs, sizeof(FloatingType) * rhs_rows, cudaHostAllocDefault);
        cudaHostAlloc(&_x, sizeof(FloatingType) * rhs_rows, cudaHostAllocDefault);
        

        for(size_t i = 0; i < rhs_rows; i++){
            _rhs[i] = 1;
        }

        return true;
    }

    template<typename FloatingType>
    bool ConjugateGradient_MultiGPUS_CUDA_NCCL<FloatingType>::save_result_to_file(const char * filename) const
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


template class LAM::ConjugateGradient_MultiGPUS_CUDA_NCCL<float>;
template class LAM::ConjugateGradient_MultiGPUS_CUDA_NCCL<double>;

}