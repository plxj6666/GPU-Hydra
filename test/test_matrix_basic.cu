#include "matrix.h"
#include <iostream>
#include <cassert>

void testGPUBasicOperations() {
    Matrix A(2, 3);
    Matrix B(2, 3);
    A.randomize();
    B.randomize();
    
    std::cout << "Matrix A:\n" << A.toString();
    std::cout << "Matrix B:\n" << B.toString();
    
    try {
        Matrix d_A = Matrix::createDeviceMatrix(2, 3);
        Matrix d_B = Matrix::createDeviceMatrix(2, 3);
        A.copyToDevice(d_A);
        B.copyToDevice(d_B);
        
        Matrix d_C = Matrix::deviceAdd(d_A, d_B);
        Matrix C(2, 3);
        C.copyFromDevice(d_C);
        
        std::cout << "Matrix C = A + B:\n" << C.toString();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

void testMatrixDimensions() {
    // 创建第一个矩阵 A (2x3)
    Matrix A(2, 3);
    uint128_t value = 0;
    
    // A[0][0]
    value = (static_cast<uint128_t>(0x0001517414516000ULL) << 64) | 
            static_cast<uint128_t>(0x41c21cb8e1170ULL);
    A.at(0,0) = FiniteField::fromParts(0, value);
    
    // A[0][1]
    value = (static_cast<uint128_t>(0x0002a2e828a2c000ULL) << 64) | 
            static_cast<uint128_t>(0x8384397dc22e0ULL);
    A.at(0,1) = FiniteField::fromParts(0, value);
    
    // A[0][2]
    value = (static_cast<uint128_t>(0x0003f45c3cf42000ULL) << 64) | 
            static_cast<uint128_t>(0xc5465730034a0ULL);
    A.at(0,2) = FiniteField::fromParts(0, value);
    
    // A[1][0]
    value = (static_cast<uint128_t>(0x0005459051458000ULL) << 64) | 
            static_cast<uint128_t>(0x0708748240660ULL);
    A.at(1,0) = FiniteField::fromParts(0, value);
    
    // A[1][1]
    value = (static_cast<uint128_t>(0x0006970465970000ULL) << 64) | 
            static_cast<uint128_t>(0x488a92948082660ULL);
    A.at(1,1) = FiniteField::fromParts(0, value);
        
    // A[1][2]
    value = (static_cast<uint128_t>(0x0007e87879e86000ULL) << 64) | 
            static_cast<uint128_t>(0x8a0cb0a6c09e0ULL);
    A.at(1,2) = FiniteField::fromParts(0, value);
    
    // 创建第二个矩阵 B (3x2)
    Matrix B(3, 2);
    
    // B[0][0]
    value = (static_cast<uint128_t>(0x0008123456789000ULL) << 64) | 
            static_cast<uint128_t>(0x123456789abcdeULL);
    B.at(0,0) = FiniteField::fromParts(0, value);
    
    // B[0][1]
    value = (static_cast<uint128_t>(0x0009987654321000ULL) << 64) | 
            static_cast<uint128_t>(0x987654321fedcULL);
    B.at(0,1) = FiniteField::fromParts(0, value);
    
    // B[1][0]
    value = (static_cast<uint128_t>(0x000aabcdef123000ULL) << 64) | 
            static_cast<uint128_t>(0xabcdef123456ULL);
    B.at(1,0) = FiniteField::fromParts(0, value);
    
    // B[1][1]
    value = (static_cast<uint128_t>(0x000b123456789000ULL) << 64) | 
            static_cast<uint128_t>(0x123456789abcULL);
    B.at(1,1) = FiniteField::fromParts(0, value);
    
    // B[2][0]
    value = (static_cast<uint128_t>(0x000cfedcba987000ULL) << 64) | 
            static_cast<uint128_t>(0xfedcba987654ULL);
    B.at(2,0) = FiniteField::fromParts(0, value);
    
    // B[2][1]
    value = (static_cast<uint128_t>(0x000d789abcdef000ULL) << 64) | 
            static_cast<uint128_t>(0x789abcdef012ULL);
    
    std::cout << "Matrix A:\n" << A.toString();
    std::cout << "Matrix B:\n" << B.toString();
    
    try {
        Matrix d_A = Matrix::createDeviceMatrix(2, 3);
        Matrix d_B = Matrix::createDeviceMatrix(3, 2);
        A.copyToDevice(d_A);
        B.copyToDevice(d_B);
        
        Matrix d_C = Matrix::deviceMultiply(d_A, d_B);
        Matrix C(2, 2);
        C.copyFromDevice(d_C);
        
        std::cout << "Matrix C = A * B:\n" << C.toString();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }
}

int main() {
    testGPUBasicOperations();
    testMatrixDimensions();
    return 0;
}