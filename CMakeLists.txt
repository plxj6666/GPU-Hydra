cmake_minimum_required(VERSION 3.10)
project(GPU_Hydra CUDA)

# CUDA设置
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# 设置XKCP路径和库
set(XKCP_DIR ${PROJECT_SOURCE_DIR}/include/XKCP)
set(XKCP_LIB ${XKCP_DIR}/bin/generic64/libXKCP.a)

# 设置Crypto++路径  
set(CRYPTOPP_DIR ${PROJECT_SOURCE_DIR}/include/crypto++)

# 包含目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${XKCP_DIR}/lib/high/Keccak
    ${XKCP_DIR}/lib/common
    ${XKCP_DIR}/lib/low/KeccakHash
    ${XKCP_DIR}/bin/generic64/libXKCP.a.headers
    ${CRYPTOPP_DIR}
)

# 链接XKCP库目录
link_directories(${XKCP_DIR}/bin/generic64)


# 源文件
file(GLOB SOURCES 
    "src/*.cu"
    "src/*.cpp"
)

# CUDA设置
set(CUDA_SEPARABLE_COMPILATION ON)

# 创建静态库
add_library(hydra STATIC ${SOURCES})
set_target_properties(hydra PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

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

# 为hydra添加XKCP依赖
target_link_libraries(hydra PRIVATE XKCP)

# 添加测试目标
add_executable(test_hydra test/test_hydra.cu)
target_link_libraries(test_hydra 
    hydra
    ${XKCP_LIB}
    cuda
    cudart
)