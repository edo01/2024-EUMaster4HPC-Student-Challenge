# Macro for adding test executables
#macro(add_test_executable TARGET_NAME SOURCE_FILE)
#    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    #target_link_libraries(${TARGET_NAME} PUBLIC LAM)
#    target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/LAM/include)
#endmacro()

add_executable(test_CPU_OMP.out "test_CG_CPU_OMP.cpp")
target_link_libraries(test_CPU_OMP.out PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(test_CPU_OMP.out PUBLIC ${CMAKE_SOURCE_DIR}/LAM/include)

add_executable(test_CPU_MPI_OMP.out "test_CG_CPU_MPI_OMP.cpp")
target_link_libraries(test_CPU_MPI_OMP.out PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_include_directories(test_CPU_MPI_OMP.out PUBLIC ${MPI_C_INCLUDE_PATH})
target_include_directories(test_CPU_MPI_OMP.out PUBLIC ${CMAKE_SOURCE_DIR}/LAM/include)


if(CUDA_FOUND)
    add_executable(test_CG_MultiGPUS_CUDA_NCCL.out "test_CG_MultiGPUS_CUDA_NCCL.cpp")
    target_link_libraries(test_CG_MultiGPUS_CUDA_NCCL.out PUBLIC LAM)
    target_link_libraries(test_CG_MultiGPUS_CUDA_NCCL.out PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)
    target_include_directories(test_CG_MultiGPUS_CUDA_NCCL.out PUBLIC ${MPI_C_INCLUDE_PATH})
    target_include_directories(test_CG_MultiGPUS_CUDA_NCCL.out PUBLIC ${CMAKE_SOURCE_DIR}/LAM/include)

    add_executable(test_CG_MultiGPUS_CUDA_MPI.out "test_CG_MultiGPUS_CUDA_MPI.cpp")
    target_link_libraries(test_CG_MultiGPUS_CUDA_MPI.out PUBLIC LAM)
    target_link_libraries(test_CG_MultiGPUS_CUDA_MPI.out PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)
    target_include_directories(test_CG_MultiGPUS_CUDA_MPI.out PUBLIC ${MPI_C_INCLUDE_PATH})
    target_include_directories(test_CG_MultiGPUS_CUDA_MPI.out PUBLIC ${CMAKE_SOURCE_DIR}/LAM/include)

    add_executable(test_CG_single_GPU.out "test_CG_single_GPU.cpp")
    target_link_libraries(test_CG_single_GPU.out PUBLIC LAM)
    target_include_directories(test_CG_single_GPU.out PUBLIC ${CMAKE_SOURCE_DIR}/LAM/include)

    add_executable(test_CG_MultiGPUS_CUDA.out "test_CG_MultiGPUS_CUDA.cpp")
    target_link_libraries(test_CG_MultiGPUS_CUDA.out PUBLIC LAM)
    target_include_directories(test_CG_MultiGPUS_CUDA.out PUBLIC ${CMAKE_SOURCE_DIR}/LAM/include)

    
endif()