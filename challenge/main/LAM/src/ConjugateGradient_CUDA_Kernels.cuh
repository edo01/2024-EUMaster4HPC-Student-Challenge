#ifndef ConjugateGradient_CUDA_Kernels_CUH
#define ConjugateGradient_CUDA_Kernels_CUH

#include <cuda.h>

constexpr int WARP_SIZE = 32;

template<typename FloatingType>
__global__ void
divide(const FloatingType * a, const FloatingType * b, FloatingType * res)
{
    *res = *a / *b;
}

template<typename FloatingType>
__device__ void
warpReduce(volatile FloatingType * a, int t)
{
    unsigned int w = WARP_SIZE;
    while (w >= 1) {
        a[t] += a[t + w];
        w >>= 1;
    }
}

template<typename FloatingType, int gridSize, int blockSize>
__global__ void
reduce(const FloatingType * a, FloatingType * sum, size_t size)
{
    // Allocate shared memory
    __shared__ FloatingType tmp[blockSize];
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

template<typename FloatingType, int gridSize, int blockSize>
__global__ void
partialDot(const FloatingType * a, const FloatingType * b, FloatingType * partialDot, size_t size)
{
    // Allocate shared memory
    __shared__ FloatingType tmp[blockSize];
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

template<typename FloatingType, int gridSize, int blockSize>
__host__ void
dot(const FloatingType, * a, const FloatingType, * b, FloatingType, * res, size_t size, cudaStream_t stream = 0)
{
    FloatingType * partialSum;
    cudaMalloc(&partialSum, sizeof(FloatingType) * gridSize);
    partialDot<FloatingType, gridSize, blockSize><<<gridSize,blockSize, 0, stream>>>(a, b, partialSum, size);
    int s = gridSize;
    while( s > 1 ){
        reduce<FloatingType, gridSize, blockSize><<<gridSize,blockSize, 0, stream>>>(partialSum, partialSum, s);
        s = (s % blockSize == 0 && s != 2) ? s/blockSize : s/blockSize + 1;
    }
    cudaMemcpyAsync(res, partialSum, sizeof(FloatingType), cudaMemcpyDeviceToDevice, stream);
    cudaFree(partialSum);
}


template<typename FloatingType, int gridSize, int blockSize>
__global__ void
axpby(const FloatingType * alpha, const FloatingType * x, const FloatingType * beta, FloatingType * y, size_t size)
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


template<typename FloatingType, int gridSize, int blockSize>
__global__ void
axpy(const FloatingType * alpha, const FloatingType * x, FloatingType * y, size_t size)
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

template<typename FloatingType, int gridSize, int blockSize>
__global__ void
minusaxpy(const FloatingType * alpha, const FloatingType * x, FloatingType * y, size_t size)
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

template<typename FloatingType, int gridSize, int blockSize>
__global__ void
xpby( const FloatingType * x, const FloatingType *beta, FloatingType * y, size_t size)
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

template<typename FloatingType, int gridSize, int blockSize>
__global__ void
gemv(FloatingType alpha, const FloatingType * A, const FloatingType * x, FloatingType beta, FloatingType * y, size_t num_rows, size_t num_cols, unsigned int stride_row)
{
    // y = alpha * A * x + beta * y

    // Allocate shared memory
    __shared__ FloatingType tmp[blockSize];
    // Retrieve ids
    unsigned int row = blockIdx.x + stride_row;
    unsigned int col = threadIdx.x;
    // Number of elements each thread has to load
    unsigned int n = (num_cols % blockSize == 0) ? num_cols / blockSize : num_cols / blockSize + 1;
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

template<typename FloatingType, int gridSize, int blockSize>
__host__ void
gemv_host(FloatingType alpha, const FloatingType * A, FloatingType * x, FloatingType beta, FloatingType * y, size_t num_rows, size_t num_cols, cudaStream_t stream = 0)
{
    unsigned int processedRow = 0;
    unsigned int numKernels = (num_rows % gridSize == 0) ? num_rows / gridSize : num_rows / gridSize + 1;
    while(numKernels > 0){
        gemv<FloatingType, gridSize,blockSize><<<gridSize, blockSize, 0, stream>>>(alpha, A, x, beta, y, num_rows, num_cols, processedRow);
        processedRow += gridSize;
        numKernels--;
    }
}

#endif //ConjugateGradient_CUDA_Kernels_CUH
