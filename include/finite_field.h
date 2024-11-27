#ifndef FINITE_FIELD_H
#define FINITE_FIELD_H

#include <cstdint>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>

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
    uint256_t value;  // 使用256位值
    
    // 辅助函数声明
    static __device__ __host__ uint256_t getP();
    static __device__ __host__ uint256_t mod256(const uint256_t& x);

public:
    // 构造函数
    __device__ __host__ FiniteField() : value() {}
    
    // 从128位高低位构造
    __host__ __device__ static FiniteField fromParts(uint128_t high, uint128_t low) {
        FiniteField result;
        result.value.high = high;
        result.value.low = low;
        return result;
    }

    
    // 获取高低位（现在是128位的高低位）
    __device__ __host__ uint128_t getHigh() const { return value.high; }
    __device__ __host__ uint128_t getLow() const { return value.low; }
    
    // 自定义输出函数
    void print(std::ostream& os) const {
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
    __device__ __host__ FiniteField& operator=(const FiniteField& other);
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
};

#endif // FINITE_FIELD_H