#ifndef MELBLAS_B_HPP
#define MELBLAS_B_HPP

#include <cstdlib>

namespace melblas
{
    template<typename FloatingType>
    class MelBLAS_B{

        public:
        
            virtual FloatingType dot(const FloatingType* x, const FloatingType* y, size_t size) const = 0;

            virtual void axpby(FloatingType alpha, const FloatingType* x, FloatingType beta, 
                                    FloatingType* y, size_t size) const = 0;

            virtual void gemv(FloatingType alpha, const FloatingType* A, const FloatingType* x,
                                    FloatingType beta, FloatingType* y, size_t rows, size_t cols) const = 0;

            virtual ~MelBLAS_B() = default;
    };
}



#endif