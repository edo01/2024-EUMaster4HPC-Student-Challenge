# Macro for adding test executables
macro(add_test_executable TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_link_libraries(${TARGET_NAME} PUBLIC LAM)
    target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/LAM/include)
endmacro()

# Add the test executables
add_test_executable(test_CPU_OMP.out "test_CG_CPU_OMP.cpp")
add_test_executable(test_CPU_MPI_OMP.out "test_CG_CPU_MPI_OMP.cpp")




#if(CUDA_FOUND)
#    add_test_executable(test_CUDA.out "test_CUDA.cpp")
#endif()