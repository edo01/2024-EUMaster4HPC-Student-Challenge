
#include "ConjugateGradient_MultiGPUS_CUDA.cuh"


namespace LAM
{
      constexpr int WARP_SIZE = 32;
      constexpr int NUM_BLOCKS=1000;
      constexpr int NUM_THREADS=1024;

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
    bool ConjugateGradient_MultiGPUS_CUDA<FloatingType>::solve( int max_iters, FloatingType rel_error)
    {
        // since only the matrix-vector multiplication is parallelized,
        // these variables are only needed in the device 0 of rank 0
        FloatingType * b_dev, * x_dev, * alpha_dev, * beta_dev, * bb_dev;
        FloatingType * rr_dev, * rr_new_dev, * r_dev, * pAp_dev, *Ap0_dev;


        // Allocate and initialize a given number of rows of A to each device
        FloatingType * A_dev[numDevices], * Ap_dev[numDevices], * p_dev[numDevices];

        // module of the rhs and of the residual on the host
        FloatingType * rr, * bb;

        rr = new FloatingType;
        bb = new FloatingType;
        int num_iters = 1;

        /**
         * ---------------------------------------------------------------
         * -----------------  Distribute the matrix  ---------------------
         * ---------------------------------------------------------------
        */
        // the i-th element contains the number of rows assigned to the i-th device
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

        #pragma omp parallel for num_threads(numDevices)
        for(int i = 0; i < numDevices; i++){
            cudaSetDevice(i);
            // contains the block i-th of the matrix A on the device
            cudaMalloc(&A_dev[i], size * (*d_size[i]) * sizeof(FloatingType));
            // contains the result of the matrix vector multiplication A * p on the device
            cudaMalloc(&Ap_dev[i],  (*d_size[i]) * sizeof(FloatingType));
            // contains the entire vector p for the matrix-vector multiplication on the device 
            cudaMalloc(&p_dev[i], size * sizeof(FloatingType));

            //copy the block i-th of the matrix A from the host to the device
            unsigned int n = (i == numDevices - 1) ? size - *d_size[i] : numRowsPerDevice * i;
            cudaMemcpyAsync(A_dev[i], A + size * n, size * (*d_size[i]) * sizeof(FloatingType), cudaMemcpyHostToDevice, streams[i]);
        }

        /*
        * ---------------------------------------------------------------
        * -----------------  data allocation  ---------------------------
        * ---------------------------------------------------------------
        */

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
        // Ap0_dev is located in device 0 and will collect all the result from the devices
        cudaMalloc(&Ap0_dev, sizeof(FloatingType) * size);
        
        // Initialize variables in device 0
        // Copy b from host to device
        cudaMemcpyAsync(b_dev, b, sizeof(FloatingType) * size, cudaMemcpyHostToDevice, streams[0]);
        // Set x to 0
        cudaMemsetAsync(x_dev, 0, sizeof(FloatingType) * size, streams[0]); // x = 0
        //set r to b
        cudaMemcpyAsync(r_dev, b_dev, sizeof(FloatingType) * size, cudaMemcpyDeviceToDevice, streams[0]); // r = b
        //set p to b
        cudaMemcpyAsync(p_dev[0], b_dev, sizeof(FloatingType) * size, cudaMemcpyDeviceToDevice, streams[0]); // p = b

        // Compute bb = b * b
        dot<FloatingType>(b_dev ,b_dev, bb_dev, size, streams[0]); // bb = b * b
        // set rr = bb
        cudaMemcpyAsync(rr_dev, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToDevice, streams[0]); // rr = bb

        /*
         * ---------------------------------------------------------------
         * -----------------  Conjugate Gradient  ------------------------
         * ---------------------------------------------------------------
         */

        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {
            
            /*
             * ---------------------------------------------------------------
             * -----------------  Matrix vector multiplication  --------------
             * ---------------------------------------------------------------
             */

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
                gemv_host<FloatingType>(1.0, A_dev[i], p_dev[i], 0.0, Ap_dev[i], *d_size[i], size, streams[i]);
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

            /*
             * -
             */

            // perform the next operations only on device 0
            cudaSetDevice(0);

            dot<FloatingType>(p_dev[0], Ap0_dev, pAp_dev, size, streams[0]);

            divide<FloatingType><<<1, 1, 0, streams[0]>>>(rr_dev, pAp_dev, alpha_dev);

            axpy<FloatingType><<<NUM_BLOCKS, NUM_THREADS, 0, streams[0]>>>(alpha_dev, p_dev[0], x_dev, size);

            minusaxpy<FloatingType><<<NUM_BLOCKS, NUM_THREADS, 0, streams[0]>>>(alpha_dev, Ap0_dev, r_dev, size);

            dot<FloatingType>(r_dev, r_dev, rr_new_dev, size, streams[0]);

            divide<FloatingType><<<1, 1, 0, streams[0]>>>(rr_new_dev, rr_dev, beta_dev);

            cudaMemcpyAsync(rr_dev, rr_new_dev, sizeof(FloatingType), cudaMemcpyDeviceToDevice, streams[0]);

            cudaMemcpyAsync(rr, rr_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost, streams[0]);
            cudaMemcpyAsync(bb, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost, streams[0]);
            cudaDeviceSynchronize();
            if(std::sqrt(*rr / *bb) < rel_error) { break; }

            xpby<FloatingType><<<NUM_BLOCKS, NUM_THREADS,0, streams[0]>>>(r_dev, beta_dev, p_dev[0], size);
        }


        cudaSetDevice(0);
        //cudaMemcpy(rr, rr_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);
        //cudaMemcpy(bb, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);

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
        x = new FloatingType[rhs_rows];

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

    ConjugateGradient_MultiGPUS_CUDA<double> cg_m_d;
    ConjugateGradient_MultiGPUS_CUDA<float> cg_m_f;

}


