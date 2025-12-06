cmake_minimum_required(VERSION 3.17)
project(set10108-cw2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(cw2
    main.cpp
    gpu_search.cu
)

target_link_libraries(cw2 PRIVATE CUDA::cudart)
set_target_properties(cw2 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 89 # RTX 4060 Laptop (Ada, SM 8.9). Adjust if building on different GPU.
)

add_custom_command(TARGET cw2 POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_SOURCE_DIR}/dataset
            $<TARGET_FILE_DIR:cw2>/dataset
    COMMENT "Copying dataset folder next to executable"
)
