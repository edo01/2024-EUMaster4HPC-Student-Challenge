#ifndef CONJUGATEGRADIENT_HPP
#define CONJUGATEGRADIENT_HPP

#include <memory>
#include <iostream>

namespace LAM
{
template<typename FloatingType>
class ConjugateGradient{
    public:
        ConjugateGradient(){
            static_assert(std::is_floating_point<FloatingType>::value, "DataType must be floating point");
        }

        /*ConjugateGradient(const char* A_filename, const char* b_filename)
        {
            static_assert(std::is_floating_point<FloatingType>::value, "DataType must be floating point");
            load_matrix_from_file(A_filename);
            load_rhs_from_file(b_filename);
        };*/
        
        bool virtual solve( int max_iters, FloatingType rel_error) = 0;

        bool virtual load_matrix_from_file(const char* filename) = 0;
        bool virtual load_rhs_from_file(const char* filename) = 0;
        bool virtual save_result_to_file(const char * filename) const = 0;
};

}
#endif