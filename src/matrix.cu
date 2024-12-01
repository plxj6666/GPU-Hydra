#include "matrix.h"
#include "polynomial.h"
#include <cuda_runtime.h>
#include <random>
#include <sstream>
#include <cstdio>
#include <iomanip>

#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 构造函数
__host__ __device__ Matrix::Matrix(int r, int c) : rows(r), cols(c), is_device(false) {
    #ifdef __CUDA_ARCH__
        // 在设备端
        data = (FiniteField*)malloc(rows * cols * sizeof(FiniteField));
        if (data == nullptr) {
            printf("Device memory allocation failed\n");  // 添加调试输出
            rows = 0;
            cols = 0;
            return;
        }
        // 添加内存初始化验证
        for (int i = 0; i < rows * cols; i++) {
            data[i] = FiniteField();
            // 验证初始化是否成功
            if (i == 0) {
                FiniteField test = data[i];
                // 验证加法操作
                FiniteField sum = test + test;
            }
        }
    #else
        // 在主机端
        cudaError_t err = cudaMallocHost(&data, rows * cols * sizeof(FiniteField));
        if (err != cudaSuccess || data == nullptr) {
            printf("Host memory allocation failed: %s\n", 
                   cudaGetErrorString(err));  // 添加错误信息
            throw std::runtime_error("Failed to allocate host memory for matrix");
        }
        // 添加内存初始化验证
        for (int i = 0; i < rows * cols; i++) {
            new (&data[i]) FiniteField();
            // 验证初始化是否成功
            if (i == 0) {
                FiniteField test = data[i];
                // 验证加法操作
                FiniteField sum = test + test;
            }
        }
    #endif
}

// 拷贝构造函数
__host__ __device__ Matrix::Matrix(const Matrix& other) : 
    rows(other.rows), cols(other.cols), is_device(other.is_device) {
    size_t size = rows * cols * sizeof(FiniteField);
    
    #ifdef __CUDA_ARCH__
        data = (FiniteField*)malloc(size);
        if (data != nullptr) {
            memcpy(data, other.data, size);
        }
    #else
        if (is_device) {
            cudaMalloc(&data, size);
        } else {
            cudaMallocHost(&data, size);
        }
        if (data != nullptr) {
            if (is_device) {
                cudaMemcpy(data, other.data, size, cudaMemcpyDeviceToDevice);
            } else {
                memcpy(data, other.data, size);
            }
        }
    #endif
}
__host__ __device__ Matrix::~Matrix() {
    if (data != nullptr) {
        #ifdef __CUDA_ARCH__
            // 在设备端
            free(data);
        #else
            // 在主机端
            if (is_device) {
                cudaFree(data);
            } else {
                cudaFreeHost(data);  // 使用cudaFreeHost替代delete[]
            }
        #endif
        data = nullptr;
    }
}

// 赋值运算符
__host__ __device__ Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        // 先释放原有内存
        if (data != nullptr) {
            #ifdef __CUDA_ARCH__
                free(data);
            #else
                if (is_device) {
                    cudaFree(data);
                } else {
                    cudaFreeHost(data);
                }
            #endif
        }
        
        rows = other.rows;
        cols = other.cols;
        is_device = other.is_device;
        
        size_t size = rows * cols * sizeof(FiniteField);
        
        #ifdef __CUDA_ARCH__
            data = (FiniteField*)malloc(size);
            if (data != nullptr) {
                memcpy(data, other.data, size);
            }
        #else
            if (is_device) {
                cudaMalloc(&data, size);
            } else {
                cudaMallocHost(&data, size);
            }
            if (data != nullptr) {
                if (is_device) {
                    cudaMemcpy(data, other.data, size, cudaMemcpyDeviceToDevice);
                } else {
                    memcpy(data, other.data, size);
                }
            }
        #endif
    }
    return *this;
}

__device__ __host__ Matrix Matrix::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        #ifdef __CUDA_ARCH__
        return Matrix();  // 在设备端返回空矩阵
        #else
        throw std::runtime_error("Matrix dimensions mismatch for multiplication");
        #endif
    }
    printf("now we have device multiply!, please change the code to use cuda kernel\n");
    Matrix result(rows, other.cols);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < other.cols; j++) {
            FiniteField sum = FiniteField::fromParts(0, 0);
            for (int k = 0; k < cols; k++) {
                sum = sum + (at(i, k) * other.at(k, j));
            }
            result.at(i, j) = sum;
        }
    }
    
    return result;
}


__host__ Matrix Matrix::power(int exponent) const {
    Matrix temp = *this;
    Matrix result = Matrix::identity(temp.getRows());
    while (exponent > 0) {
        if (exponent % 2 == 1) {
            result = multiplyMatrices(result, temp);  // 调用优化的矩阵乘法
        }
        temp = multiplyMatrices(temp, temp);
        exponent >>= 1;
    }
    return result;
}

__host__ FiniteField Matrix::trace() const {
    if (rows != cols) {
        #ifdef __CUDA_ARCH__
        return FiniteField::fromParts(0, 0);
        #else
        throw std::runtime_error("Matrix must be square to compute trace");
        #endif
    }

    FiniteField trace = FiniteField::fromParts(0, 0);

    // 累加主对角线元素
    for (int i = 0; i < rows; ++i) {
        trace = (trace + at(i, i)).mod(); // 每次累加后取模
    }

    return trace;
}

// 在设备上创建矩阵
__host__ Matrix Matrix::createDeviceMatrix(int rows, int cols) {
    Matrix matrix(rows, cols);
    CHECK_CUDA_ERROR(cudaMalloc(&matrix.data, rows * cols * sizeof(FiniteField)));
    matrix.is_device = true;
    return matrix;
}

// GPU矩阵加法
__host__ Matrix Matrix::deviceAdd(const Matrix& A, const Matrix& B) {
    if (A.cols != B.cols || A.rows != B.rows) {
        throw std::runtime_error("Matrix dimensions do not match");
    }
    
    if (!A.is_device || !B.is_device) {
        throw std::runtime_error("Input matrices must be on device");
    }
    
    Matrix C = createDeviceMatrix(A.rows, A.cols);
    
    Matrix *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc(&d_A, sizeof(Matrix)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_B, sizeof(Matrix)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_C, sizeof(Matrix)));
    
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, &A, sizeof(Matrix), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, &B, sizeof(Matrix), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_C, &C, sizeof(Matrix), cudaMemcpyHostToDevice));
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (A.getCols() + blockSize.x - 1) / blockSize.x,
        (A.getRows() + blockSize.y - 1) / blockSize.y
    );
    
    matrixAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return C;
}


// 内存拷贝方法
__host__ void Matrix::copyToDevice(Matrix& d_matrix) const {
    size_t size = rows * cols * sizeof(FiniteField);
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix.data, data, size, cudaMemcpyHostToDevice));
}

__host__ void Matrix::copyFromDevice(const Matrix& d_matrix) {
    if (!d_matrix.is_device) {
        throw std::runtime_error("Source matrix is not a device matrix");
    }
    
    if (d_matrix.rows != rows || d_matrix.cols != cols) {
        throw std::runtime_error("Matrix dimensions do not match for copy");
    }
    
    size_t size = rows * cols * sizeof(FiniteField);
    CHECK_CUDA_ERROR(cudaMemcpy(data, d_matrix.data, size, cudaMemcpyDeviceToHost));
}

// 随机化矩阵元素
__host__ void Matrix::randomize() {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis;
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            uint128_t random_value = dis(gen);
            random_value = (random_value << 64) | dis(gen);
            at(i, j) = FiniteField::fromParts(0, random_value);
        }
    }
}

// 转换为字符串
__host__ std::string Matrix::toString() const {
    std::ostringstream oss;
    
    // 计算每个元素的最大宽度
    size_t maxWidth = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::ostringstream temp;
            data[i * cols + j].printToStream(temp);
            maxWidth = std::max(maxWidth, temp.str().length());
        }
    }
    
    // 打印矩阵
    oss << "\n";  // 先换行，让输出更清晰
    for (int i = 0; i < rows; ++i) {
        oss << "[ ";  // 每行开始
        for (int j = 0; j < cols; ++j) {
            std::ostringstream temp;
            data[i * cols + j].printToStream(temp);
            std::string value = temp.str();
            
            // 右对齐并填充空格
            oss << std::setw(maxWidth) << value;
            
            // 在元素之间添加分隔符，最后一个元素后不添加
            if (j < cols - 1) {
                oss << " | ";
            }
        }
        oss << " ]" << std::endl;  // 每行结束
    }
    return oss.str();
}

// 添加一个辅助函数来验证设备内存
__host__ bool Matrix::validateDevicePointer() const {
    cudaPointerAttributes attrs;
    return cudaPointerGetAttributes(&attrs, data) == cudaSuccess && 
           attrs.type == cudaMemoryTypeDevice;
}

__host__ __device__ bool Matrix::isInvertible() const {
    if (rows != cols) {
        return false;  // 非方阵一定不可逆
    }    
    FiniteField det = determinant();
    return !(det == FiniteField::fromParts(0,0));
}

__device__ __host__ FiniteField Matrix::determinant() const {
    if (rows != cols) {
        return FiniteField::fromParts(0,0);
    }
    
    // 1x1到4x4矩阵原直接计算方式
    if (rows <= 4) {
        return determinantSmall();
    }
    
    // 对于更大的矩阵，使用LU分解
    return determinantLU();
}

__device__ __host__ FiniteField Matrix::determinantSmall() const {
    if (rows == 1) return at(0, 0);
    if (rows == 2) return at(0,0) * at(1,1) - at(0,1) * at(1,0);
    
    if (rows == 3) {
        FiniteField det;
        // 正项
        det = det + (at(0,0) * at(1,1) * at(2,2));
        det = det + (at(0,1) * at(1,2) * at(2,0));
        det = det + (at(0,2) * at(1,0) * at(2,1));
        // 负项
        det = det - (at(0,2) * at(1,1) * at(2,0));
        det = det - (at(0,1) * at(1,0) * at(2,2));
        det = det - (at(0,0) * at(1,2) * at(2,1));
        return det;
    }
    
    if (rows == 4) {
        FiniteField det;
        for (int j = 0; j < 4; j++) {
            FiniteField minor_det;
            if (j == 0) {
                minor_det = (at(1,1) * at(2,2) * at(3,3) + 
                           at(1,2) * at(2,3) * at(3,1) + 
                           at(1,3) * at(2,1) * at(3,2)) -
                          (at(1,3) * at(2,2) * at(3,1) + 
                           at(1,2) * at(2,1) * at(3,3) + 
                           at(1,1) * at(2,3) * at(3,2));
            } else if (j == 1) {
                minor_det = (at(1,0) * at(2,2) * at(3,3) + 
                           at(1,2) * at(2,3) * at(3,0) + 
                           at(1,3) * at(2,0) * at(3,2)) -
                          (at(1,3) * at(2,2) * at(3,0) + 
                           at(1,2) * at(2,0) * at(3,3) + 
                           at(1,0) * at(2,3) * at(3,2));
                minor_det = -minor_det;
            } else if (j == 2) {
                minor_det = (at(1,0) * at(2,1) * at(3,3) + 
                           at(1,1) * at(2,3) * at(3,0) + 
                           at(1,3) * at(2,0) * at(3,1)) -
                          (at(1,3) * at(2,1) * at(3,0) + 
                           at(1,1) * at(2,0) * at(3,3) + 
                           at(1,0) * at(2,3) * at(3,1));
            } else {
                minor_det = (at(1,0) * at(2,1) * at(3,2) + 
                           at(1,1) * at(2,2) * at(3,0) + 
                           at(1,2) * at(2,0) * at(3,1)) -
                          (at(1,2) * at(2,1) * at(3,0) + 
                           at(1,1) * at(2,0) * at(3,2) + 
                           at(1,0) * at(2,2) * at(3,1));
                minor_det = -minor_det;
            }
            det = det + (at(0,j) * minor_det);
        }
        return det;
    }
    
    return FiniteField::fromParts(0,0);
}

__device__ __host__ FiniteField Matrix::determinantLU() const {
    // 使用栈内存代替设备内存
    FiniteField L_data[64];  // 支持最大8x8矩阵
    FiniteField U_data[64];
    
    // 初化L和U
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            int idx = i * cols + j;
            if(i == j) {
                L_data[idx] = FiniteField::fromParts(0,1);
            } else {
                L_data[idx] = FiniteField::fromParts(0,0);
            }
            U_data[idx] = FiniteField::fromParts(0,0);
        }
    }
    
    // LU分解
    for(int i = 0; i < rows; i++) {
        // 计算U的第i行
        for(int j = i; j < cols; j++) {
            FiniteField sum = FiniteField::fromParts(0,0);
            for(int k = 0; k < i; k++) {
                sum = sum + L_data[i * cols + k] * U_data[k * cols + j];
            }
            U_data[i * cols + j] = at(i,j) - sum;
        }
        
        // 检查对线素是否为0
        if(U_data[i * cols + i].isZero()) {
            return FiniteField::fromParts(0,0); // 矩阵奇
        }
        
        // 计算L的第i列
        FiniteField u_ii_inv = U_data[i * cols + i].inverse();
        for(int j = i + 1; j < rows; j++) {
            FiniteField sum = FiniteField::fromParts(0,0);
            for(int k = 0; k < i; k++) {
                sum = sum + L_data[j * cols + k] * U_data[k * cols + i];
            }
            L_data[j * cols + i] = (at(j,i) - sum) * u_ii_inv;
        }
    }
    
    // 计算行列式：对角线元素的乘
    FiniteField det = FiniteField::fromParts(0,1);
    for(int i = 0; i < rows; i++) {
        det = det * U_data[i * cols + i];
    }
    
    return det;
}

__host__ Polynomial Matrix::characteristicPolynomial() const {
        printf("Host: Starting characteristic polynomial calculation\n");
        const int n = rows;
        Polynomial charPoly;
        
        FiniteField c[MAX_MATRIX_SIZE + 1];
        FiniteField p[MAX_MATRIX_SIZE + 1];
        
        for (int i = 0; i <= n; ++i) {
            c[i] = FiniteField::fromParts(0, 0);
            p[i] = FiniteField::fromParts(0, 0);
        }
        
        c[n] = FiniteField::fromParts(0, 1);
        
        for (int k = 1; k <= n; ++k) {
            printf("Host: Computing power %d\n", k);
            Matrix Ak = this->power(k);
            printf("Host: Matrix power %d computed, calculating trace\n", k);
            p[k] = Ak.trace().mod();
            printf("Host: Trace for power %d = ", k);
            p[k].print();
            printf("\n");
            
            FiniteField sum = FiniteField::fromParts(0, 0);
            for (int j = 1; j < k; ++j) {
                sum = (sum + c[n - j] * p[k - j]).mod();
            }
            c[n - k] = ((-p[k] - sum) / FiniteField::fromParts(0, k)).mod();
            printf("Host: Coefficient c[%d] = ", n-k);
            c[n - k].print();
            printf("\n");
        }
        
        for (int i = 0; i <= n; ++i) {
            charPoly.setCoefficient(i, c[i]);
        }
        
        return charPoly;
}




__host__ Polynomial Matrix::minimalPolynomial() const {
    // 首先计特征多项式
    Polynomial charPoly = characteristicPolynomial();
    
    // 计算导数
    Polynomial derivative;
    for(int i = 1; i <= charPoly.degree(); i++) {
        if(i % 2 == 1) {  // 在特征为2的域中，只有奇次项的导数非零
            derivative.setCoefficient(i-1, charPoly[i]);
        }
    }
    
    // 计算GCD并迭代消除重复因子
    Polynomial minPoly = charPoly;
    Polynomial gcd = minPoly.gcd(derivative);
    
    // 检查 gcd 是否为常数多项式（degree = 0）且数为1
    while(gcd.degree() > 0 || (gcd.degree() == 0 && !(gcd[0] == FiniteField::fromParts(0, 1)))) {
        minPoly = minPoly / gcd;
        derivative = Polynomial();
        for(int i = 1; i <= minPoly.degree(); i++) {
            if(i % 2 == 1) {
                derivative.setCoefficient(i-1, minPoly[i]);
            }
        }
        gcd = minPoly.gcd(derivative);
    }
    
    minPoly.normalize();
    return minPoly;
}

__host__ Matrix Matrix::multiplyMatrices(const Matrix& A, const Matrix& B) const {
    // 分配设备内存：Matrix对象
    Matrix *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeof(Matrix));
    cudaMalloc(&d_B, sizeof(Matrix));
    cudaMalloc(&d_C, sizeof(Matrix));

    // 分配设备内存：矩阵数据
    FiniteField *d_A_data, *d_B_data, *d_C_data;
    int A_size = A.getRows() * A.getCols() * sizeof(FiniteField);
    int B_size = B.getRows() * B.getCols() * sizeof(FiniteField);
    int C_size = A.getRows() * B.getCols() * sizeof(FiniteField);
    cudaMalloc(&d_A_data, A_size);
    cudaMalloc(&d_B_data, B_size);
    cudaMalloc(&d_C_data, C_size);

    // 初始化结果矩阵的数据为0
    cudaMemset(d_C_data, 0, C_size);

    // 复制输入矩阵的数据到设备
    cudaMemcpy(d_A_data, A.getData(), A_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_data, B.getData(), B_size, cudaMemcpyHostToDevice);

    // 创建设备端Matrix对象的主机副本
    Matrix h_A = A;
    Matrix h_B = B;
    Matrix h_C(A.getRows(), B.getCols());
    
    // 更新data指针为设备内存地址
    h_A.data = d_A_data;
    h_B.data = d_B_data;
    h_C.data = d_C_data;

    // 复制Matrix对象到设备（包含了rows、cols和更新后的data指针）
    cudaMemcpy(d_A, &h_A, sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, &h_B, sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, &h_C, sizeof(Matrix), cudaMemcpyHostToDevice);
    
    // 设置网格和块的维度
    int rows = A.getRows();
    dim3 blockSize(rows, rows);
    dim3 gridSize(1, 1);
    
    // 调用核函数
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C);
    
    // 检查并同步
    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    // 创建结果矩阵
    Matrix C(A.getRows(), B.getCols());
    cudaMemcpy(C.data, d_C_data, C_size, cudaMemcpyDeviceToHost);
    
    // 释放设备内存
    cudaFree(d_A_data);
    cudaFree(d_B_data);
    cudaFree(d_C_data);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return C;
}


// CUDA核函数 - 矩阵加法
__global__ void matrixAddKernel(const Matrix* A, const Matrix* B, Matrix* C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < A->getCols() && idy < A->getRows()) {
        C->at(idy, idx) = A->at(idy, idx) + B->at(idy, idx);
    }
}


__global__ void matrixMultiplyKernel(const Matrix* A, const Matrix* B, Matrix* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A->getRows() || col >= B->getCols()) return; // 早期返回，防止越界访问
    FiniteField sum = FiniteField::fromParts(0, 0);

    for (int k = 0; k < A->getCols(); k++) {
        FiniteField a = A->at(row, k);
        FiniteField b = B->at(k, col);
        FiniteField prod = a * b;
        sum = sum + prod;
    }
    C->at(row, col) = sum;
}