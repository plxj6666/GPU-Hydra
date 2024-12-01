#ifndef MATRIX_H
#define MATRIX_H

#include "finite_field.h"
#include "polynomial.h"
#include <cassert>
#include <cusolverDn.h>
#include <cublas_v2.h>

#define MAX_MATRIX_SIZE 8  // 最大矩阵大小

class Matrix {
private:
    int rows;
    int cols;
    FiniteField* data;
    bool is_device;
    __device__ __host__ FiniteField determinantSmall() const;
    __device__ __host__ FiniteField determinantLU() const;

public:
    // 构造和析构函数
    __host__ __device__ Matrix(int rows, int cols);
    __host__ __device__ Matrix(const Matrix& other);
    __host__ __device__ ~Matrix();
    __host__ __device__ Matrix& operator=(const Matrix& other);
    __host__ __device__ Matrix() : rows(0), cols(0), data(nullptr), is_device(false) {}

    // 基本访问方法
    __host__ __device__ int getRows() const { return rows; }
    __host__ __device__ int getCols() const { return cols; }
    __host__ __device__ FiniteField* getData() const{ return data; }
    __host__ __device__ inline FiniteField& at(int i, int j) {
        assert(i >= 0 && i < rows);
        assert(j >= 0 && j < cols);
        return data[i * cols + j];
    }
    __host__ __device__ inline const FiniteField& at(int i, int j) const {
        assert(i >= 0 && i < rows);
        assert(j >= 0 && j < cols);
        return data[i * cols + j];
    }

    // 基本运算
    __device__ __host__ Matrix operator+(const Matrix& other) const;
    __device__ __host__ Matrix operator*(const Matrix& other) const;
    __device__ __host__ Matrix operator*(const FiniteField& scalar) const;
    __device__ __host__ FiniteField determinant() const;
    __host__ Matrix multiplyMatrices(const Matrix& A, const Matrix& B) const;

    // GPU相关操作
    __host__ static Matrix createDeviceMatrix(int rows, int cols);
    __host__ static Matrix deviceAdd(const Matrix& A, const Matrix& B);
    __host__ void copyToDevice(Matrix& d_matrix) const;
    __host__ void copyFromDevice(const Matrix& d_matrix);
    __host__ bool validateDevicePointer() const;

    // 矩阵属性和操作
    __host__ __device__ bool isInvertible() const;
    __host__ void randomize();
    __host__ std::string toString() const;


    // 静态方法
    __host__ __device__ static Matrix identity(int size) {
        Matrix result(size, size);
        for (int i = 0; i < size; i++) {
            result.at(i, i) = FiniteField::fromParts(0, 1);
        }
        return result;
    }
    __host__ __device__ static Matrix zero(int rows, int cols) {
        Matrix result(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.at(i, j) = FiniteField::fromParts(0, 0);
            }
        }
        return result;
    }

    // 检查矩阵是否为零矩阵
    __device__ __host__ bool isZero() const {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                if (!at(i, j).isZero()) {
                    return false;
                }
            }
        }
        return true;
    }

    // 计算特征多项式
    __host__ Polynomial characteristicPolynomial() const;
    __host__ Matrix power(int k) const;
    __host__ FiniteField trace() const;
    // 计算最小多项式
    __host__ Polynomial minimalPolynomial() const;
};

// CUDA核函数声明
__global__ void matrixAddKernel(const Matrix* A, const Matrix* B, Matrix* C);
__global__ void matrixMultiplyKernel(const Matrix* A, const Matrix* B, Matrix* C);
#endif // MATRIX_H