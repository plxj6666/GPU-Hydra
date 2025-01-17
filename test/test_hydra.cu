#include <iostream>
#include "Hydra.h"
#include <cuda_runtime.h>

int main() {
    // 初始化参数
    uint128_t p_ = uint128_t(0x8000000000000000ULL) << 64 | uint128_t(0x000000000000002dULL);
    FiniteField p = FiniteField::fromParts(0, p_);
    int t = 5;   // 明文长度
    int sec = 128; // 安全级别

    // 主机端初始化 Hydra
    Hydra h_hydra(p, t, sec);

    // 将 Hydra 复制到设备
    Hydra* d_hydra = h_hydra.copyToDevice();

    // 初始化主机端输入数据
    FiniteFieldArray h_state_in(t);
    for (int i = 0; i < t; ++i) {
        h_state_in[i] = FiniteField::fromParts(0, i); // 输入为 0, 1, 2, ...
    }

    // 在设备端分配输出数组
    FiniteFieldArray* d_state_out;
    cudaMalloc(&d_state_out, sizeof(FiniteFieldArray));   
    FiniteField* d_elements;
    cudaMalloc(&d_elements, sizeof(FiniteField) * t); // 为 elements 分配设备内存
    FiniteFieldArray h_state_out(t); // 在主机端初始化
    h_state_out.setElements(d_elements, false);      // 设置 elements 指针（不拥有内存）

    cudaMemcpy(d_state_out, &h_state_out, sizeof(FiniteFieldArray), cudaMemcpyHostToDevice);

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start);

    // 调用加密核函数
    hydraEncrypt<<<1, 1>>>(d_state_out, d_hydra, t);

    // 记录结束时间
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // 计算时间差
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "Time taken for hydraEncrypt: " << milliseconds << " ms" << std::endl;

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}