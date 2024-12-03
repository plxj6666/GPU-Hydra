#include "../include/Hydra.h"
// 在主机端调用的包装函数
// 在主机端调用的包装函数
// 添加CUDA错误检查宏
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

__host__ FiniteFieldArray launchHydraEncrypt(int t, int sec) {
    // 1. 初始化主机端Hydra对象
    uint128_t p_ = uint128_t(0x8000000000000000ULL) << 64 | uint128_t(0x000000000000002dULL);
    FiniteField p = FiniteField::fromParts(0, p_);
    Hydra h_hydra(p, t, sec);

    // 2. 为设备端Hydra对象分配内存
    Hydra* d_hydra;
    CUDA_CHECK(cudaMalloc(&d_hydra, sizeof(Hydra)));

    // 3. 处理矩阵Me、Mi、Mh的深拷贝
    Matrix d_Me = h_hydra.getME();
    Matrix d_Mi = h_hydra.getMI();
    Matrix d_Mh = h_hydra.getMH();

    // 为矩阵的elements分配设备内存并复制数据
    FiniteField* d_Me_elements;
    CUDA_CHECK(cudaMalloc(&d_Me_elements, d_Me.getRows() * d_Me.getCols() * sizeof(FiniteField)));
    CUDA_CHECK(cudaMemcpy(d_Me_elements, d_Me.getElements(), 
        d_Me.getRows() * d_Me.getCols() * sizeof(FiniteField), cudaMemcpyHostToDevice));
    d_Me.setElements(d_Me_elements);

    // 对Mi和Mh执行相同操作
    FiniteField* d_Mi_elements;
    CUDA_CHECK(cudaMalloc(&d_Mi_elements, d_Mi.getRows() * d_Mi.getCols() * sizeof(FiniteField)));
    CUDA_CHECK(cudaMemcpy(d_Mi_elements, d_Mi.getElements(), 
        d_Mi.getRows() * d_Mi.getCols() * sizeof(FiniteField), cudaMemcpyHostToDevice));
    d_Mi.setElements(d_Mi_elements);

    FiniteField* d_Mh_elements;
    CUDA_CHECK(cudaMalloc(&d_Mh_elements, d_Mh.getRows() * d_Mh.getCols() * sizeof(FiniteField)));
    CUDA_CHECK(cudaMemcpy(d_Mh_elements, d_Mh.getElements(), 
        d_Mh.getRows() * d_Mh.getCols() * sizeof(FiniteField), cudaMemcpyHostToDevice));
    d_Mh.setElements(d_Mh_elements);

    // 4. 处理rc_b数组的深拷贝
    size_t rc_b_size = h_hydra.getRE1() + h_hydra.getRE2() + h_hydra.getRI();
    FiniteFieldArray* d_rc_b;
    CUDA_CHECK(cudaMalloc(&d_rc_b, rc_b_size * sizeof(FiniteFieldArray)));
    
    // 为每个FiniteFieldArray分配内存并复制数据
    for (size_t i = 0; i < rc_b_size; i++) {
        FiniteField* d_rc_b_elements;
        size_t array_size = h_hydra.getRCB()[i].getSize();
        CUDA_CHECK(cudaMalloc(&d_rc_b_elements, array_size * sizeof(FiniteField)));
        CUDA_CHECK(cudaMemcpy(d_rc_b_elements, h_hydra.getRCB()[i].getElements(), 
            array_size * sizeof(FiniteField), cudaMemcpyHostToDevice));
        
        FiniteFieldArray temp_array(array_size);
        temp_array.setElements(d_rc_b_elements, false);  // 不接管内存所有权
        CUDA_CHECK(cudaMemcpy(&d_rc_b[i], &temp_array, sizeof(FiniteFieldArray), cudaMemcpyHostToDevice));
    }

    // 5. 处理rc_h数组的深拷贝
    size_t rc_h_size = h_hydra.getRH();
    FiniteFieldArray* d_rc_h;
    CUDA_CHECK(cudaMalloc(&d_rc_h, rc_h_size * sizeof(FiniteFieldArray)));
    
    for (size_t i = 0; i < rc_h_size; i++) {
        FiniteField* d_rc_h_elements;
        size_t array_size = h_hydra.getRCH()[i].getSize();
        CUDA_CHECK(cudaMalloc(&d_rc_h_elements, array_size * sizeof(FiniteField)));
        CUDA_CHECK(cudaMemcpy(d_rc_h_elements, h_hydra.getRCH()[i].getElements(), 
            array_size * sizeof(FiniteField), cudaMemcpyHostToDevice));
        
        FiniteFieldArray temp_array(array_size);
        temp_array.setElements(d_rc_h_elements, false);  // 不接管内存所有权
        CUDA_CHECK(cudaMemcpy(&d_rc_h[i], &temp_array, sizeof(FiniteFieldArray), cudaMemcpyHostToDevice));
    }

    // 6. 创建临时Hydra对象并设置设备端数据
    Hydra temp_hydra = h_hydra;
    temp_hydra.getME().setDeviceMatrix(d_Me);
    temp_hydra.getMI().setDeviceMatrix(d_Mi);
    temp_hydra.getMH().setDeviceMatrix(d_Mh);
    temp_hydra.setRCB(d_rc_b);
    temp_hydra.setRCH(d_rc_h);
    
    // 设置标志表明这些是设备指针
    temp_hydra.setDevicePointers(true);

    // 7. 将完整的Hydra对象复制到设备
    CUDA_CHECK(cudaMemcpy(d_hydra, &temp_hydra, sizeof(Hydra), cudaMemcpyHostToDevice));

    // 8. 为输出分配内存
    FiniteFieldArray* d_output;
    CUDA_CHECK(cudaMalloc(&d_output, sizeof(FiniteFieldArray)));
    
    FiniteField* d_output_elements;
    CUDA_CHECK(cudaMalloc(&d_output_elements, t * sizeof(FiniteField)));
    
    FiniteFieldArray temp_output(t);
    temp_output.setElements(d_output_elements, false); // 不接管内存所有权
    CUDA_CHECK(cudaMemcpy(d_output, &temp_output, sizeof(FiniteFieldArray), cudaMemcpyHostToDevice));

    // 9. 启动核函数
    hydraEncrypt<<<1, 1>>>(d_output, d_hydra, t);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // 确保设备计算完成

    // 10. 获取结果
    FiniteField* h_output_elements = new FiniteField[t];
    CUDA_CHECK(cudaMemcpy(h_output_elements, d_output_elements, t * sizeof(FiniteField), cudaMemcpyDeviceToHost));

    FiniteFieldArray h_output(t);
    h_output.setElements(h_output_elements, true); // 主机端接管内存所有权

    // 11. 清理设备内存
    cudaFree(d_Me_elements);
    cudaFree(d_Mi_elements);
    cudaFree(d_Mh_elements);
    
    // 清理rc_b和rc_h的内存
    for (size_t i = 0; i < rc_b_size; i++) {
        FiniteFieldArray temp;
        cudaMemcpy(&temp, &d_rc_b[i], sizeof(FiniteFieldArray), cudaMemcpyDeviceToHost);
        cudaFree(temp.getElements());
    }
    cudaFree(d_rc_b);

    for (size_t i = 0; i < rc_h_size; i++) {
        FiniteFieldArray temp;
        cudaMemcpy(&temp, &d_rc_h[i], sizeof(FiniteFieldArray), cudaMemcpyDeviceToHost);
        cudaFree(temp.getElements());
    }
    cudaFree(d_rc_h);
    
    cudaFree(d_output_elements);
    cudaFree(d_output);
    cudaFree(d_hydra);

    return h_output;
}

__host__ void testHydraCPU(int t, int sec) {
    // 初始化参数
    uint128_t p_ = uint128_t(0x8000000000000000ULL) << 64 | uint128_t(0x000000000000002dULL);
    FiniteField p = FiniteField::fromParts(0, p_);
    
    // 创建 Hydra 实例
    Hydra hydra(p, t, sec);
    
    // 创建测试输入
    FiniteFieldArray state_in(t);
    for (int i = 0; i < t; ++i) {
        state_in[i] = FiniteField::fromParts(0, i);
    }
    
    // 创建测试密钥和参数
    FiniteFieldArray MK(4);  // 密钥
    for (int i = 0; i < 4; ++i) {
        MK[i] = FiniteField::fromParts(0, 0);
    }
    
    FiniteFieldArray IV(3);  // 初始化向量
    for (int i = 0; i < 3; ++i) {
        IV[i] = FiniteField::fromParts(0, 1);
    }
    
    FiniteFieldArray N(1);   // Nonce
    N[0] = FiniteField::fromParts(0, 2);
    
    // 打印初始状态
    printf("\nInitial state:\n");
    for (int i = 0; i < t; ++i) {
        printf("state_in[%d] = ", i);
        state_in[i].print();
        printf("\n");
    }
    
    // 打印矩阵信息
    printf("\nMatrix Me info:\n");
    printf("Rows: %zu, Cols: %zu\n", hydra.getME().getRows(), hydra.getME().getCols());
    
    // 尝试执行矩阵乘法
    printf("\nTesting matrix multiplication:\n");
    FiniteFieldArray test_state(4);
    for (int i = 0; i < 4; ++i) {
        test_state[i] = FiniteField::fromParts(0, i);
    }
    
    FiniteFieldArray result = hydra.getME() * test_state;
    printf("Matrix multiplication result:\n");
    for (int i = 0; i < result.getSize(); ++i) {
        printf("result[%d] = ", i);
        result[i].print();
        printf("\n");
    }
    
    // 打印rc_b信息
    printf("\nrc_b info:\n");
    for (int i = 0; i < hydra.getRE1(); ++i) {
        printf("rc_b[%d] size: %zu\n", i, hydra.getRCB()[i].getSize());
        for (int j = 0; j < hydra.getRCB()[i].getSize(); ++j) {
            printf("rc_b[%d][%d] = ", i, j);
            hydra.getRCB()[i][j].print();
            printf("\n");
        }
    }
    
    try {
        // 尝试加密
        printf("\nTrying encryption:\n");
        FiniteFieldArray result = hydra.encrypt(state_in, MK, IV, N);
        
        // 打印结果
        printf("\nEncryption result:\n");
        for (int i = 0; i < result.getSize(); ++i) {
            printf("result[%d] = ", i);
            result[i].print();
            printf("\n");
        }
    } catch (const std::exception& e) {
        printf("Exception caught: %s\n", e.what());
    }
}

// 在main函数中调用
int main() {
    // testHydraCPU(5, 128);
    FiniteFieldArray result = launchHydraEncrypt(5, 128);
    for (int i = 0; i < result.getSize(); ++i) {
        printf("result[%d] = ", i);
        result[i].print();
        printf("\n");
    }
    return 0;
}