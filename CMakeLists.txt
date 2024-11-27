cmake_minimum_required(VERSION 3.10)
project(GPU_Hydra CUDA)

# 设置CUDA架构
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 添加include目录
include_directories(${PROJECT_SOURCE_DIR}/include)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -Xcompiler -fPIC")

# 创建库文件
add_library(gpu_hydra STATIC
    src/finite_field.cu
    src/matrix.cu
    src/Hydra.cu
    src/polynomial.cu
)

# 设置库的属性
set_target_properties(gpu_hydra PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# 添加所有测试可执行文件
set(TEST_FILES
    test_matrix_basic
    test_finite_field
    test_unit128
    test_matrix_isinvertible  
    test_polynomial
)

# 为每个测试文件创建可执行文件
foreach(test ${TEST_FILES})
    add_executable(${test} test/${test}.cu)
    set_target_properties(${test} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(${test} PRIVATE gpu_hydra)
endforeach()

# 编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()

# 在 CMakeLists.txt 中添加调试标志
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
