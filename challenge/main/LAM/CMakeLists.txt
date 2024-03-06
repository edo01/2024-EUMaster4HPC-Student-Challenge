# Add the library LAM
file(GLOB_RECURSE SOURCES "src/*.cpp")

if(CUDA_FOUND)
    # Add CUDA source files to the SOURCES variable
    file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

    list(APPEND SOURCES ${CUDA_SOURCES})
    
    # Add NCCl include and library directories 
    set(NCCL_INCLUDE_DIRS "/apps/USE/easybuild/release/2023.1/software/NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.2.0/include")
    set(NCCL_LIBRARIES "/apps/USE/easybuild/release/2023.1/software/NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.2.0/lib/libnccl.so")
endif()

# Create LAM target
add_library(LAM STATIC ${SOURCES})

# Link NCCL to your target
target_link_libraries(LAM PUBLIC ${NCCL_LIBRARIES})
# Include NCCL directories
target_include_directories(LAM PUBLIC ${NCCL_INCLUDE_DIRS})

# Link libraries to the executable
#target_link_libraries(LAM PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Additional settings for MPI
#target_include_directories(LAM PUBLIC ${MPI_C_INCLUDE_PATH})