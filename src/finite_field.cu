#include "finite_field.h"
#include <cstdint>
#include <iostream>
#include <iomanip>

// 获取模数P (返回256位值)
__device__ __host__ uint256_t FiniteField::getP() {
    uint256_t p;
    p.high = 0;  // 高128位为0
    
    // 设置低128位
    // 0x8000000000000000000000000000002d
    uint128_t high64 = uint128_t(0x8000000000000000ULL);
    uint128_t low64 = uint128_t(0x000000000000002dULL);
    p.low = (high64 << 64) | low64;
    
    return p;
}

// 运算符实现
__device__ __host__ FiniteField FiniteField::operator+(const FiniteField& other) const {
    // 添加调试信息
    #ifndef __CUDA_ARCH__
    if (this == nullptr) {
        printf("Error: null this pointer in operator+\n");
        return FiniteField();
    }
    #endif
    
    FiniteField result;
    result.value.low = value.low + other.value.low;
    result.value.high = value.high + other.value.high;
    return result;
}

__device__ __host__ FiniteField FiniteField::operator-(const FiniteField& other) const {
    FiniteField result;
    uint256_t p = getP();
    
    if (value >= other.value) {
        // 直接相减
        result.value.low = value.low - other.value.low;
        result.value.high = value.high - other.value.high - (value.low < other.value.low ? 1 : 0);
    } else {
        // 需要加上模数再减
        uint256_t temp;
        temp.low = p.low - other.value.low;
        temp.high = p.high - other.value.high - (p.low < other.value.low ? 1 : 0);
        result.value.low = temp.low + value.low;
        result.value.high = temp.high + value.high + (result.value.low < temp.low ? 1 : 0);
    }
    return result;
}

__device__ __host__ FiniteField FiniteField::operator*(const FiniteField& other) const {
    uint64_t a1 = static_cast<uint64_t>(value.low >> 64);
    uint64_t a0 = static_cast<uint64_t>(value.low);
    uint64_t b1 = static_cast<uint64_t>(other.value.low >> 64);
    uint64_t b0 = static_cast<uint64_t>(other.value.low);
    
    uint256_t result(0, 0);
    
    // 1. a0 * b0
    uint128_t prod00 = static_cast<uint128_t>(a0) * b0;
    result.low = prod00;
    
    // 2. 交叉项 (a0 * b1 + a1 * b0)
    uint128_t prod01 = static_cast<uint128_t>(a0) * b1;
    uint128_t prod10 = static_cast<uint128_t>(a1) * b0;
    uint128_t cross = prod01 + prod10;
    
    // 处理cross相加的进位
    if (cross < prod01) {
        result.high += 1;
    }
    
    // 将交叉项结果左移64位并加到结果中
    uint128_t temp = result.low;  // 保存原始值
    result.low += (cross << 64);
    // 如果新的low小于原始值，说明产生了进位
    if (result.low < temp) {
        result.high += 1;
    }
    result.high += (cross >> 64);
    
    // 3. a1 * b1
    uint128_t prod11 = static_cast<uint128_t>(a1) * b1;
    result.high += prod11;
    
    // #ifndef __CUDA_ARCH__
    // std::cout << "Debug: intermediate result before mod = ";
    // std::cout << "High: 0x" << std::hex << static_cast<uint64_t>(result.high >> 64)
    //           << std::setw(16) << std::setfill('0') << static_cast<uint64_t>(result.high)
    //           << ", Low: 0x" << std::setw(16) << std::setfill('0') << static_cast<uint64_t>(result.low >> 64)
    //           << std::setw(16) << std::setfill('0') << static_cast<uint64_t>(result.low) << std::endl;
    // #endif
    
    // 最后进行模运算
    result = mod256(result);
    
    FiniteField final_result;
    final_result.value = result;
    return final_result;
}

// 256位模运算的实现
__device__ __host__ uint256_t FiniteField::mod256(const uint256_t& x) {
    uint256_t p = getP();
    uint256_t result = x;
    
    // 如果结果小于p，直接返回
    if (!(result >= p)) {
        return result;
    }
    
    // 由于p是128位的，处理高位部分
    while (result.high > 0) {
        // 将result.high和p.low都拆分成64位
        uint64_t q1 = static_cast<uint64_t>(result.high >> 64);
        uint64_t q0 = static_cast<uint64_t>(result.high);
        uint64_t p1 = static_cast<uint64_t>(p.low >> 64);
        uint64_t p0 = static_cast<uint64_t>(p.low);
        
        // 计算q * p，注意处理进位
        uint128_t prod00 = static_cast<uint128_t>(q0) * p0;
        uint128_t prod01 = static_cast<uint128_t>(q0) * p1;
        uint128_t prod10 = static_cast<uint128_t>(q1) * p0;
        uint128_t prod11 = static_cast<uint128_t>(q1) * p1;
        
        // 组合结果
        uint256_t sub;
        sub.low = prod00;
        sub.high = prod11;
        
        // 处理中间项
        uint128_t middle = prod01 + prod10;
        bool carry = (middle < prod01);
        
        // 将middle左移64位并加到结果中
        if (sub.low + (middle << 64) < sub.low) {
            sub.high += 1;
        }
        sub.low += (middle << 64);
        sub.high += (middle >> 64);
        if (carry) {
            sub.high += (uint128_t(1) << 64);
        }
        
        // 执行减法
        if (result.low < sub.low) {
            result.high--;
        }
        result.low -= sub.low;
        result.high -= sub.high;
    }
    
    // 最后检查是否需要再减一次p
    if (result.low >= p.low) {
        result.low -= p.low;
    }
    
    return result;
}

__device__ __host__ bool FiniteField::operator==(const FiniteField& other) const {
    return value.high == other.value.high && value.low == other.value.low;
}

__device__ __host__ FiniteField& FiniteField::operator=(const FiniteField& other) {
    if (this != &other) {
        value = other.value;
    }
    return *this;
}

__device__ __host__ FiniteField FiniteField::operator-() const {
    if (value.high == 0 && value.low == 0) return *this;
    uint256_t p = getP();
    FiniteField result;
    result.value.low = p.low - value.low;
    result.value.high = p.high - value.high - (p.low < value.low ? 1 : 0);
    return result;
}

__device__ __host__ FiniteField FiniteField::inverse() const {
    if (isZero()) {
        #ifdef __CUDA_ARCH__
        return FiniteField::fromParts(0, 0);
        #else
        throw std::runtime_error("Division by zero");
        #endif
    }
    
    // 使用费马小定理：a^(p-2) mod p
    uint256_t p = getP();
    uint256_t exp;
    exp.high = 0;
    exp.low = p.low - 2;  // p-2
    
    // 快速幂算法
    FiniteField result = FiniteField::fromParts(0, 1);
    FiniteField base = *this;
    
    while (exp.low != 0 || exp.high != 0) {
        if (exp.low & 1) {
            result = result * base;
        }
        base = base * base;
        // 右移1位
        exp.low = (exp.low >> 1) | (exp.high << 127);
        exp.high >>= 1;
    }
    
    return result;
}

__device__ __host__ FiniteField FiniteField::operator/(const FiniteField& other) const {
    #ifdef __CUDA_ARCH__
    if (other.isZero()) {
        return FiniteField::fromParts(0, 0);  // 在设备端返回0
    }
    #else
    if (other.isZero()) {
        throw std::runtime_error("Division by zero");
    }
    #endif
    
    return (*this) * other.inverse();
}

__host__ __device__ FiniteField FiniteField::mod() const {
    FiniteField result;
    result.value = mod256(value);
    return result;
}