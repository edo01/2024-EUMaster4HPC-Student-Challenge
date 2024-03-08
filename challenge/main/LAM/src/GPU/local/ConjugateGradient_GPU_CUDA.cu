#include "ConjugateGradient_GPU_CUDA.cuh"


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
    __host__ void dot(const FloatingType* a, const FloatingType* b, FloatingType* res, size_t size)
    {
        FloatingType * partialSum;
        cudaMalloc(&partialSum, sizeof(FloatingType) * NUM_BLOCKS);
        partialDot<FloatingType><<<NUM_BLOCKS,NUM_THREADS>>>(a, b, partialSum, size);
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
    __host__ void gemv_host(FloatingType alpha, const FloatingType * A, FloatingType * x, FloatingType beta, FloatingType * y, size_t num_rows, size_t num_cols)
    {
        unsigned int processedRow = 0;
        unsigned int numKernels = (num_rows % NUM_BLOCKS == 0) ? num_rows / NUM_BLOCKS : num_rows / NUM_BLOCKS + 1;
        while(numKernels > 0){
            gemv<FloatingType><<<NUM_BLOCKS, NUM_THREADS>>>(alpha, A, x, beta, y, num_rows, num_cols, processedRow);
            processedRow += NUM_BLOCKS;
            numKernels--;
        }
    }
    
    template<typename FloatingType>
    bool ConjugateGradient_GPU_CUDA<FloatingType>::solve( int max_iters, FloatingType rel_error) {

        // since only the matrix-vector multiplication is parallelized,
        // these variables are only needed in the device 0 of rank 0
        FloatingType *Ap_dev, *p_dev, *A_dev, * b_dev, * x_dev, * alpha_dev, * beta_dev, * bb_dev;
        FloatingType * rr_dev, * rr_new_dev, * r_dev, * pAp_dev;

        // module of the rhs and of the residual on the host
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

        dot<FloatingType>(b_dev ,b_dev, bb_dev, size); // bb = b * b
        cudaMemcpy(rr_dev, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToDevice); // rr = bb

        for(num_iters = 1; num_iters <= max_iters; num_iters++)
        {

            gemv_host<FloatingType>(1.0, A_dev, p_dev, 0.0, Ap_dev, size, size);

            dot<FloatingType>(p_dev, Ap_dev, pAp_dev, size);

            divide<FloatingType><<<1, 1>>>(rr_dev, pAp_dev, alpha_dev);

            axpy<FloatingType><<<NUM_BLOCKS, NUM_THREADS>>>(alpha_dev, p_dev, x_dev, size);

            minusaxpy<FloatingType><<<NUM_BLOCKS, NUM_THREADS>>>(alpha_dev, Ap_dev, r_dev, size);

            dot<FloatingType>(r_dev, r_dev, rr_new_dev, size);

            divide<FloatingType><<<1, 1>>>(rr_new_dev, rr_dev, beta_dev);

            cudaMemcpy(rr_dev, rr_new_dev, sizeof(FloatingType), cudaMemcpyDeviceToDevice);

            cudaMemcpy(rr, rr_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);
            cudaMemcpy(bb, bb_dev, sizeof(FloatingType), cudaMemcpyDeviceToHost);
            if(std::sqrt(*rr / *bb) < rel_error) { break; }

            xpby<FloatingType><<<NUM_BLOCKS, NUM_THREADS>>>(r_dev, beta_dev, p_dev, size);
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
        x = new FloatingType[rhs_rows];

        fread(b, sizeof(FloatingType), rhs_rows, file);

        fclose(file);

        return true;
    }

    template<typename FloatingType>
    bool ConjugateGradient_GPU_CUDA<FloatingType>::save_result_to_file(const char * filename) const
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

    ConjugateGradient_GPU_CUDA<double> gpu1;
    //ConjugateGradient_GPU_CUDA<float> gpu2;
}