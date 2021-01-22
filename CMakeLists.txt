cmake_minimum_required(VERSION 3.17)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.2/bin/nvcc)
project(hamming CUDA)

set(CMAKE_CUDA_STANDARD 14)
add_executable(hamming.exe hamming.cu)
set_target_properties(
        hamming
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)