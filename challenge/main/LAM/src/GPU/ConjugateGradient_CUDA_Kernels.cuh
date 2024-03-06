#ifndef ConjugateGradient_CUDA_Kernels_CUH
#define ConjugateGradient_CUDA_Kernels_CUH

namespace LAM {

    constexpr int WARP_SIZE = 32;

    template<typename FloatingType>
    __global__ void divide(const FloatingType * a, const FloatingType * b, FloatingType * res);

    template<typename FloatingType>
    __device__ void warpReduce(volatile FloatingType * a, int t);

    template<typename FloatingType, int gridSize, int blockSize>
    __global__ void reduce(const FloatingType * a, FloatingType * sum, size_t size);

    template<typename FloatingType, int gridSize, int blockSize>
    __global__ void partialDot(const FloatingType * a, const FloatingType * b, FloatingType * partialDot, size_t size);

    template<typename FloatingType, int gridSize, int blockSize>
    __host__ void dot(const FloatingType* a, const FloatingType* b, FloatingType* res, size_t size, cudaStream_t stream = 0);

    template<typename FloatingType, int gridSize, int blockSize>
    __global__ void axpby(const FloatingType * alpha, const FloatingType * x, const FloatingType * beta, FloatingType * y, size_t size);

    template<typename FloatingType, int gridSize, int blockSize>
    __global__ void axpy(const FloatingType * alpha, const FloatingType * x, FloatingType * y, size_t size);

    template<typename FloatingType, int gridSize, int blockSize>
    __global__ void minusaxpy(const FloatingType * alpha, const FloatingType * x, FloatingType * y, size_t size);

    template<typename FloatingType, int gridSize, int blockSize>
    __global__ void xpby( const FloatingType * x, const FloatingType *beta, FloatingType * y, size_t size);

    template<typename FloatingType, int gridSize, int blockSize>
    __global__ void gemv(FloatingType alpha, const FloatingType * A, const FloatingType * x, FloatingType beta, FloatingType * y, size_t num_rows, size_t num_cols, unsigned int stride_row);

    template<typename FloatingType, int gridSize, int blockSize>
    __host__ void gemv_host(FloatingType alpha, const FloatingType * A, FloatingType * x, FloatingType beta, FloatingType * y, size_t num_rows, size_t num_cols, cudaStream_t stream = 0);

} // namespace LAM

#endif //ConjugateGradient_CUDA_Kernels_CUH
