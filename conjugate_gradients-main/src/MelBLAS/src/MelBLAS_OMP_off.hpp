#ifndef MelBLAS_OMP_off_HPP
#define MelBLAS_OMP_off_HPP

#ifdef _OPENMP
#include <omp.h>
#endif

#include "MelBLAS_B.hpp"

namespace melblas
{

    template<typename FloatingType>
    class MelBLAS_OMP_off :
    public MelBLAS_B<FloatingType>{

        public:
            FloatingType dot(const FloatingType* x, const FloatingType* y, size_t size) const;

            void axpby(FloatingType alpha, const FloatingType* x, FloatingType beta,
                        FloatingType* y, size_t size) const;

            void gemv(FloatingType alpha, const FloatingType* A, const FloatingType* x, FloatingType beta,
                        FloatingType* y, size_t rows, size_t cols) const;

            ~MelBLAS_OMP_off() = default;
    };


    template<typename FloatingType>
    FloatingType MelBLAS_OMP_off<FloatingType>::dot(const FloatingType* x, const FloatingType* y, size_t size) const
    {
        FloatingType result = 0.0;

        #pragma omp parallel for reduction(+:result)
        for(size_t i = 0; i < size; i++)
        {
            result += x[i] * y[i];
        }
        return result;
    }

    template<typename FloatingType>
    void MelBLAS_OMP_off<FloatingType>::axpby(FloatingType alpha, const FloatingType* x, FloatingType beta,
                                                FloatingType* y, size_t size) const
    {
        // y = alpha * x + beta * y

        //#pragma omp parallel for
        #pragma omp target
        for(size_t i = 0; i < size; i++)
        {
            y[i] = alpha * x[i] + beta * y[i];
        }
    }

    template<typename FloatingType>
    void MelBLAS_OMP_off<FloatingType>::gemv(FloatingType alpha, const FloatingType* A,
                    const FloatingType* x, FloatingType beta, FloatingType* y, size_t rows, size_t cols) const
    {
        // y = alpha * A * x + beta * y;
        #pragma omp target teams distribute
        for(size_t r = 0; r < rows; r++)
        {
            FloatingType y_val = 0.0;
            for(size_t c = 0; c < cols; c++)
            {
                y_val += alpha * A[r * cols + c] * x[c];
            }
            y[r] = beta * y[r] + y_val;
        }
    }


}


#endif