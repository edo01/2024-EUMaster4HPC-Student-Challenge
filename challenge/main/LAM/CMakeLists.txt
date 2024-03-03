# Add the library LAM
file(GLOB_RECURSE SOURCES "src/*.cpp")

if(CUDA_FOUND)
    # Add CUDA source files to the SOURCES variable
    file(GLOB_RECURSE CUDA_SOURCES "src/*.cu")

    list(APPEND SOURCES ${CUDA_SOURCES})
endif()

# Create LAM target
add_library(LAM SHARED ${SOURCES})

# Link libraries to the executable
target_link_libraries(LAM PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)

# Additional settings for MPI
target_include_directories(LAM PUBLIC ${MPI_C_INCLUDE_PATH})