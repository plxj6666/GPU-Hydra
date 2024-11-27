#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include "finite_field.h"
#include <cassert>

class Polynomial {
private:
    static const int MAX_DEGREE = 64;  // 最大多项式度数
    FiniteField coefficients[MAX_DEGREE + 1];  // 系数数组
    int deg;  // 实际度数

    // 私有辅助函数
    __device__ __host__ void updateDegree();  // 更新多项式实际次数
    
public:
    // 构造函数
    __device__ __host__ Polynomial();  // 零多项式
    __device__ __host__ Polynomial(const FiniteField& constant);  // 常数多项式
    __device__ __host__ Polynomial(const Polynomial& other);  // 拷贝构造
    
    // 基本访问方法
    __device__ __host__ int degree() const { return deg; }
    __device__ __host__ bool isZero() const { return deg == -1; }
    __device__ __host__ FiniteField leadingCoefficient() const;
    
    // 系数访问
    __device__ __host__ const FiniteField& operator[](int i) const;
    __device__ __host__ FiniteField& operator[](int i);
    __device__ __host__ void setCoefficient(int power, const FiniteField& value);
    
    // 基本算术运算
    __device__ __host__ Polynomial operator+(const Polynomial& other) const;
    __device__ __host__ Polynomial operator-(const Polynomial& other) const;
    __device__ __host__ Polynomial operator*(const Polynomial& other) const;
    __device__ __host__ Polynomial operator/(const Polynomial& other) const;
    __device__ __host__ Polynomial operator%(const Polynomial& other) const;
    
    // 规范化和模运算
    __device__ __host__ void normalize();  // 使最高次项系数为1
    __device__ __host__ void modInField();  // 确保所有系数在有限域中
    
    // 特殊操作
    __device__ __host__ bool isIrreducible() const;  // 判断是否不可约
    __device__ __host__ Polynomial gcd(const Polynomial& other) const;  // 最大公约式
    
    // 赋值运算符
    __device__ __host__ Polynomial& operator=(const Polynomial& other);
};

#endif // POLYNOMIAL_H
