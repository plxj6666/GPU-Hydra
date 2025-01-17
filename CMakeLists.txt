cmake_minimum_required(VERSION 3.10)
project(GPU_Hydra CUDA)

# CUDA设置
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# 设置XKCP路径
set(XKCP_DIR ${PROJECT_SOURCE_DIR}/include/XKCP)

# 设置Crypto++路径
set(CRYPTOPP_DIR ${PROJECT_SOURCE_DIR}/include/crypto++)

# 包含目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${XKCP_DIR}/lib/high/Keccak
    ${XKCP_DIR}/lib/common
    ${XKCP_DIR}/lib/low/KeccakHash
    ${XKCP_DIR}/bin/generic64/libXKCP.so.headers
    ${CRYPTOPP_DIR}
)

# 源文件
file(GLOB SOURCES 
    "src/*.cu"
    "src/*.cpp"
)

# 创建静态库
add_library(hydra STATIC ${SOURCES})

# 编译选项
target_compile_options(hydra PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 
        --use_fast_math
        -extended-lambda 
        --expt-relaxed-constexpr
        --device-c
    >
)

# 设置输出目录
set_target_properties(hydra PROPERTIES 
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
)

# 链接外部库
target_link_libraries(hydra
    ${XKCP_DIR}/bin/generic64/libXKCP.so
    crypto++
)