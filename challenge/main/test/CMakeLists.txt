# Macro for adding test executables
#macro(add_test_executable TARGET_NAME SOURCE_FILE)
#    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    #target_link_libraries(${TARGET_NAME} PUBLIC LAM)
#    target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/LAM/include)
#endmacro()

# Add the test executables
#add_test_executable(test_CPU_OMP.out "test_CG_CPU_OMP.cpp")
#add_test_executable(test_CPU_MPI_OMP.out "test_CG_CPU_MPI_OMP.cpp")
#add_test_executable(test_CPU_OACC.out "test_CG_CPU_OACC.cpp")

add_executable(test_CPU_OMP.out "test_CG_CPU_OMP.cpp")
target_link_libraries(test_CPU_OMP.out PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(test_CPU_OMP.out PUBLIC ${CMAKE_SOURCE_DIR}/LAM/include)

add_executable(test_CPU_MPI_OMP.out "test_CG_CPU_MPI_OMP.cpp")
target_link_libraries(test_CPU_MPI_OMP.out PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_include_directories(test_CPU_MPI_OMP.out PUBLIC ${MPI_C_INCLUDE_PATH})
target_include_directories(test_CPU_MPI_OMP.out PUBLIC ${CMAKE_SOURCE_DIR}/LAM/include)

#add_executable(test_CPU_OACC.out "test_CG_CPU_OACC.cpp")
#target_compile_options(test_CPU_OACC.out PRIVATE -fopenacc)
#target_include_directories(test_CPU_OACC.out PUBLIC ${CMAKE_SOURCE_DIR}/LAM/include)


#link openacc in the same fashion of openmp
#target_link_libraries(test_CPU_OACC.out PRIVATE OpenACC::OpenACC_CXX)
#target_link_libraries(test_CPU_OACC.out PRIVATE gomp -lm -lgomp)

#target_compile_options(test_CPU_OACC.out PRIVATE -fopenacc)
#target_link_libraries(test_CPU_OACC.out PRIVATE gomp -lm -lgomp)
#target_compile_options(test_CPU_OACC.out PRIVATE)


#if(CUDA_FOUND)
#    add_test_executable(test_CUDA.out "test_CUDA.cpp")
#endif()