#ifndef FINITE_FIELD_H
#define FINITE_FIELD_H

#include <cstdint>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cassert>
#ifdef __CUDA_ARCH__
using uint128_t = unsigned __int128;
#else
using uint128_t = __uint128_t;
#endif

// 256位整数结构体
struct uint256_t {
    uint128_t high;
    uint128_t low;
    
    __device__ __host__ uint256_t() : high(0), low(0) {}
    __device__ __host__ uint256_t(uint128_t h, uint128_t l) : high(h), low(l) {}
    __device__ __host__ explicit uint256_t(uint128_t x) : high(0), low(x) {}
    
    // 添加基本运算
    __device__ __host__ uint256_t operator+(const uint256_t& other) const {
        uint256_t result;
        result.low = low + other.low;
        result.high = high + other.high + (result.low < low ? 1 : 0);
        return result;
    }
    __device__ __host__ bool operator>=(const uint256_t& other) const {
        return (high > other.high) || (high == other.high && low >= other.low);
    }

};

class FiniteField {
private:
    // 辅助函数声明
    static __device__ __host__ uint256_t getP();
    static __device__ __host__ uint256_t mod256(const uint256_t& x);

public:
    // 添加和修改构造函数
    __device__ __host__ FiniteField() { value.high = 0; value.low = 0; }
    __device__ __host__ FiniteField(const FiniteField& other) { value = other.value; }
    __device__ __host__ FiniteField(const uint256_t& v) : value(v) {}

    // 确保赋值运算符实现正确
    __device__ __host__ FiniteField& operator=(const FiniteField& other) {
        if (this != &other) {
            value = other.value;
        }
        return *this;
    }

    // 确保value类型定义正确
    uint256_t value;
    // 从128位高低位构造
    __host__ __device__ static FiniteField fromParts(uint128_t high, uint128_t low) {
        FiniteField result;
        result.value.high = high;
        result.value.low = low;
        return result;
    }

    // 获取模数P
    static __device__ __host__ FiniteField getFiniteFieldModule() {
        FiniteField result;
        result.value = getP();
        return result;
    }

    static __device__ __host__ uint256_t getModulevalue() {
        return getP();
    }
    // 获取高低位（现在是128位的高低位）
    __device__ __host__ uint128_t getHigh() const { return value.high; }
    __device__ __host__ uint128_t getLow() const { return value.low; }
    __device__ __host__ uint256_t getValue() const { return value; }
    // 添加设备端打印方法
    __device__ __host__ void print() const {
        #ifdef __CUDA_ARCH__
        // 设备端打印
        printf("0x%016llx%016llx%016llx%016llx",
            static_cast<unsigned long long>(value.high >> 64),
            static_cast<unsigned long long>(value.high),
            static_cast<unsigned long long>(value.low >> 64),
            static_cast<unsigned long long>(value.low));
        #else
        // 主机端打印
        std::cout << "0x" 
                  << std::hex << std::setw(16) << std::setfill('0') 
                  << static_cast<uint64_t>(value.high >> 64)
                  << std::setw(16) << std::setfill('0') 
                  << static_cast<uint64_t>(value.high)
                  << std::setw(16) << std::setfill('0') 
                  << static_cast<uint64_t>(value.low >> 64)
                  << std::setw(16) << std::setfill('0') 
                  << static_cast<uint64_t>(value.low);
        #endif
    }
    
    // 原来的 print 方法改名为 printToStream
    void printToStream(std::ostream& os) const {
        uint64_t high_high = static_cast<uint64_t>(value.high >> 64);
        uint64_t high_low = static_cast<uint64_t>(value.high);
        uint64_t low_high = static_cast<uint64_t>(value.low >> 64);
        uint64_t low_low = static_cast<uint64_t>(value.low);
        
        os << "High: 0x" << std::hex << std::setw(16) << std::setfill('0') << high_high
           << std::setw(16) << std::setfill('0') << high_low
           << ", Low: 0x" << std::hex << std::setw(16) << std::setfill('0') << low_high
           << std::setw(16) << std::setfill('0') << low_low;
    }
    
    // 运算符声明
    __device__ __host__ FiniteField operator+(const FiniteField& other) const;
    __device__ __host__ FiniteField operator*(const FiniteField& other) const;
    __device__ __host__ FiniteField operator-(const FiniteField& other) const;
    __device__ __host__ FiniteField operator-() const;
    __device__ __host__ FiniteField operator/(const FiniteField& other) const;
    __device__ __host__ bool operator==(const FiniteField& other) const;
    __device__ __host__ bool operator<(const FiniteField& other) const;
    __device__ __host__ bool operator!=(const FiniteField& other) const;
    __device__ __host__ FiniteField operator<<(int shift) const;
    __device__ __host__ FiniteField operator|(const FiniteField& other) const;
    friend std::ostream& operator<<(std::ostream& os, const FiniteField& value);
    __host__ __device__ FiniteField mod() const;  // 新增的公共方法
    // 计算乘法逆元
    __device__ __host__ FiniteField inverse() const;
    __device__ __host__ bool isZero() const {return value.high == 0 && value.low == 0;}
    __device__ __host__ bool isNegative() const {
        uint256_t half_p;
        half_p.high = 0;
        half_p.low = getP().low / 2;
        return value.low > half_p.low;
    }
    __device__ __host__ bool isOne() const;  // 判断是否为1
    __device__ __host__ FiniteField abs() const;  // 返回绝对值
    __device__ __host__ static FiniteField power(const FiniteField& base, uint64_t exponent);
};

// 有限域元素数组类
class FiniteFieldArray {
private:
    FiniteField* elements;
    size_t size;
    bool owns_memory;  // 标记是否需要在析构时释放内存

public:
    // 构造函数
    __host__ __device__ FiniteFieldArray() : elements(nullptr), size(0), owns_memory(false) {}
    
    __host__ __device__ FiniteFieldArray(size_t n) : size(n), owns_memory(true) {
        if (n > 0) {
            #ifdef __CUDA_ARCH__
            cudaMalloc(&elements, sizeof(FiniteField) * n);
            #else
            elements = new FiniteField[n];
            #endif
        } else {
            elements = nullptr;
        }
    }


    // 析构函数
    __host__ __device__ ~FiniteFieldArray() {
        if (owns_memory && elements != nullptr) {
            #ifdef __CUDA_ARCH__
            cudaFree(elements);
            #else
            delete[] elements;
            #endif
        }
    }


    // 只保留一个版本的 setElements
    __host__ __device__ void setElements(FiniteField* new_elements, bool take_ownership = false) {
        if (owns_memory && elements != nullptr) {
            #ifdef __CUDA_ARCH__
            delete[] elements;
            #else
            delete[] elements;
            #endif
        }
        elements = new_elements;
        owns_memory = take_ownership;
    }
    
    // 添加拷贝构造函数
    __host__ __device__ FiniteFieldArray(const FiniteFieldArray& other) : size(other.size), owns_memory(true) {
        elements = new FiniteField[size];
        for (size_t i = 0; i < size; ++i) {
            elements[i] = other.elements[i];
        }
    }

    // 添加赋值运算符
    __host__ __device__ FiniteFieldArray& operator=(const FiniteFieldArray& other) {
        if (this != &other) {
            // 释放原有内存
            if (elements) {
                delete[] elements;
            }
            
            // 分配新内存并复制数据
            size = other.size;
            elements = new FiniteField[size];
            for (size_t i = 0; i < size; ++i) {
                elements[i] = other.elements[i];
            }
        }
        return *this;
    }

    // 添加获取和设置elements的方法
    __host__ __device__ FiniteField* getElements() const { return elements; }
    
    
    // 添加更详细的调试打印
    __device__ void debugPrint() const {
        printf("FiniteFieldArray debug: size=%zu, elements=%p\n", size, elements);
        if (elements != nullptr) {
            printf("First element: ");
            elements[0].print();
            printf("\n");
        }
    }

    // 数组访问运算符
    __host__ __device__ FiniteField& operator[](size_t index) {
        #ifndef __CUDA_ARCH__
        if (index >= size) {
            throw std::out_of_range("Index out of range");
        }
        #endif
        return elements[index];
    }

    __device__ __host__ FiniteFieldArray operator+(const FiniteFieldArray& other) const {
        assert(size == other.getSize());
        FiniteFieldArray result(size);
        for (int i = 0; i < size; ++i) {
            result[i] = elements[i] + other[i];
        }
        return result;
    };

    __device__ __host__ FiniteFieldArray operator-(const FiniteFieldArray& other) const {
        assert(size == other.getSize());
        FiniteFieldArray result(size);
        for (int i = 0; i < size; ++i) {
            result[i] = elements[i] - other[i];
        }
        return result;
    };
    // const 版本的数组访问运算符
    __host__ __device__ const FiniteField& operator[](size_t index) const {
        #ifndef __CUDA_ARCH__
        if (index >= size) {
            throw std::out_of_range("Index out of range");
        }
        #endif
        return elements[index];
    }
    __device__ __host__ FiniteFieldArray mod() const {
        FiniteFieldArray result(size);
        for (int i = 0; i < size; ++i) {
            result[i] = elements[i].mod();
        }
        return result;
    }
    
    // 获取数组大小
    __host__ __device__ size_t getSize() const {
        return size;
    }

    // 添加 setElement 方法
    __host__ __device__ void setElement(size_t index, const FiniteField& value) {
        #ifndef __CUDA_ARCH__
        if (index >= size) {
            throw std::out_of_range("Index out of range");
        }
        #endif
        elements[index] = value;
    }
};

#endif // FINITE_FIELD_H