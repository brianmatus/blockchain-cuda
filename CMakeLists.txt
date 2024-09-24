cmake_minimum_required(VERSION 3.29)
project(blockchain_cuda CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

include_directories(/opt/cuda/include)
link_directories(/opt/cuda/lib64)

#find_package(CUDA REQUIRED)

set (SOURCES
        main.cu
        elements/Block.cu
        elements/Blockchain.cu
        utils/sha256.cu
)

add_executable(blockchain_cuda ${SOURCES})

set_target_properties(blockchain_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)


#TODO  Needed?
# Link the C++ standard library
target_link_libraries(blockchain_cuda
        PRIVATE
        stdc++
)