#ifndef LinearAlgebraMeluxina_HPP
#define LinearAlgebraMeluxina_HPP

#include "../src/ConjugateGradient.hpp"
#include "../src/CPU/ConjugateGradient_CPU_OMP.hpp"
#include "../src/CPU/ConjugateGradient_CPU_MPI_OMP.hpp"

#ifdef USE_CUDA
#include "../src/GPU/distributed/ConjugateGradient_MultiGPUS_CUDA_NCCL.cuh"
#include "../src/GPU/local/ConjugateGradient_GPU_CUDA.cuh"
#include "../src/GPU/local/ConjugateGradient_MultiGPUS_CUDA.cuh"
#endif


#endif // LinearAlgebraMeluxina_HPP