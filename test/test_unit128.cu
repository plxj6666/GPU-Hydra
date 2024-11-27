#include <stdio.h>
#include <cuda_runtime.h>

// 尝试定义 uint128
#ifdef __CUDA_ARCH__
using uint128_t = unsigned __int128;
#endif

__global__ void testUint128Kernel() {
    #ifdef __CUDA_ARCH__
        uint128_t test = 1;
        printf("GPU: uint128 可用\n");
    #endif
}

int main() {
    printf("开始测试 uint128 支持...\n");
    
    testUint128Kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // 检查是否有错误
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA错误: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    return 0;
}