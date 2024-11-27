#include "polynomial.h"

// 构造函数实现
__device__ __host__ Polynomial::Polynomial() : deg(-1) {
    for (int i = 0; i <= MAX_DEGREE; i++) {
        coefficients[i] = FiniteField::fromParts(0, 0);
    }
}

__device__ __host__ Polynomial::Polynomial(const FiniteField& constant) : deg(0) {
    coefficients[0] = constant;
    for (int i = 1; i <= MAX_DEGREE; i++) {
        coefficients[i] = FiniteField::fromParts(0, 0);
    }
    if (constant.isZero()) {
        deg = -1;
    }
}

__device__ __host__ Polynomial::Polynomial(const Polynomial& other) : deg(other.deg) {
    for (int i = 0; i <= MAX_DEGREE; i++) {
        coefficients[i] = other.coefficients[i];
    }
}

// 赋值运算符
__device__ __host__ Polynomial& Polynomial::operator=(const Polynomial& other) {
    if (this != &other) {
        deg = other.deg;
        for (int i = 0; i <= MAX_DEGREE; i++) {
            coefficients[i] = other.coefficients[i];
        }
    }
    return *this;
}

// 更新多项式次数
__device__ __host__ void Polynomial::updateDegree() {
    deg = MAX_DEGREE;
    while (deg >= 0 && coefficients[deg].isZero()) {
        deg--;
    }
}

// 访问方法
__device__ __host__ FiniteField Polynomial::leadingCoefficient() const {
    if (deg == -1) return FiniteField::fromParts(0, 0);
    return coefficients[deg];
}

__device__ __host__ const FiniteField& Polynomial::operator[](int i) const {
    assert(i >= 0 && i <= MAX_DEGREE);
    return coefficients[i];
}

__device__ __host__ FiniteField& Polynomial::operator[](int i) {
    assert(i >= 0 && i <= MAX_DEGREE);
    return coefficients[i];
}

__device__ __host__ void Polynomial::setCoefficient(int power, const FiniteField& value) {
    assert(power >= 0 && power <= MAX_DEGREE);
    coefficients[power] = value;
    if (power > deg && !value.isZero()) {
        deg = power;
    } else if (power == deg && value.isZero()) {
        updateDegree();
    }
}

// 基本算术运算
__device__ __host__ Polynomial Polynomial::operator+(const Polynomial& other) const {
    Polynomial result;
    int max_deg = (deg > other.deg) ? deg : other.deg;
    
    for (int i = 0; i <= max_deg; i++) {
        FiniteField sum = (i <= deg ? coefficients[i] : FiniteField::fromParts(0, 0)) +
                         (i <= other.deg ? other.coefficients[i] : FiniteField::fromParts(0, 0));
        if (!sum.isZero() || i <= max_deg) {
            result.coefficients[i] = sum;
        }
    }
    
    result.updateDegree();
    return result;
}

__device__ __host__ Polynomial Polynomial::operator-(const Polynomial& other) const {
    Polynomial result;
    int max_deg = (deg > other.deg) ? deg : other.deg;
    
    for (int i = 0; i <= max_deg; i++) {
        FiniteField diff = (i <= deg ? coefficients[i] : FiniteField::fromParts(0, 0)) -
                          (i <= other.deg ? other.coefficients[i] : FiniteField::fromParts(0, 0));
        if (!diff.isZero() || i <= max_deg) {
            result.coefficients[i] = diff;
        }
    }
    
    result.updateDegree();
    return result;
}

__device__ __host__ Polynomial Polynomial::operator*(const Polynomial& other) const {
    if (deg == -1 || other.deg == -1) return Polynomial();
    
    Polynomial result;
    for (int i = 0; i <= deg; i++) {
        for (int j = 0; j <= other.deg; j++) {
            if (i + j <= MAX_DEGREE) {
                result.coefficients[i + j] = result.coefficients[i + j] +
                    coefficients[i] * other.coefficients[j];
            }
        }
    }
    
    result.updateDegree();
    return result;
}

// 除法运算
__device__ __host__ Polynomial Polynomial::operator/(const Polynomial& other) const {
    assert(!other.isZero());  // 除数不能为零
    
    if (deg < other.deg) {
        return Polynomial();  // 商为0
    }
    
    Polynomial quotient;
    Polynomial remainder = *this;
    
    // 主除法循环
    while (remainder.deg >= other.deg && !remainder.isZero()) {
        // 计算当前步骤的商的次数和系数
        int power = remainder.deg - other.deg;
        FiniteField coeff = remainder.coefficients[remainder.deg] / other.coefficients[other.deg];
        
        // 构造当前步骤的商项
        Polynomial term;
        term.deg = power;
        term.coefficients[power] = coeff;
        
        // 更新商和余数
        quotient = quotient + term;
        remainder = remainder - (term * other);
    }
    
    return quotient;
}

// 取模运算
__device__ __host__ Polynomial Polynomial::operator%(const Polynomial& other) const {
    assert(!other.isZero());  // 除数不能为零
    
    if (deg < other.deg) {
        return *this;  // 直接返回被除数
    }
    
    Polynomial remainder = *this;
    
    // 主除法循环
    while (remainder.deg >= other.deg && !remainder.isZero()) {
        // 计算当前步骤的商的次数和系数
        int power = remainder.deg - other.deg;
        FiniteField coeff = remainder.coefficients[remainder.deg] / other.coefficients[other.deg];
        
        // 构造当前步骤的商项
        Polynomial term;
        term.deg = power;
        term.coefficients[power] = coeff;
        
        // 更新余数
        remainder = remainder - (term * other);
    }
    
    return remainder;
}

// 规范化（使最高次项系数为1）
__device__ __host__ void Polynomial::normalize() {
    if (deg == -1) return;  // 零多项式不需要规范化
    
    FiniteField leading = coefficients[deg];
    if (leading.isZero()) {
        updateDegree();
        if (deg != -1) {
            normalize();
        }
        return;
    }
    
    // 计算首项系数的逆元
    FiniteField inv = leading.inverse();
    
    // 将所有系数乘以逆元
    for (int i = 0; i <= deg; i++) {
        coefficients[i] = coefficients[i] * inv;
    }
}

// 确保所有系数在有限域中
__device__ __host__ void Polynomial::modInField() {
    for (int i = 0; i <= deg; i++) {
        coefficients[i] = coefficients[i].mod();
    }
    updateDegree();
}

// 计算最大公约式
__device__ __host__ Polynomial Polynomial::gcd(const Polynomial& other) const {
    if (other.isZero()) {
        Polynomial result = *this;
        result.normalize();
        return result;
    }
    
    return other.gcd(*this % other);
}

// 判断是否不可约
__device__ __host__ bool Polynomial::isIrreducible() const {
    if (deg <= 1) return true;  // 0次和1次多项式都是不可约的
    
    // 使用有限域上的不可约性测试
    Polynomial x;
    x.deg = 1;
    x.coefficients[1] = FiniteField::fromParts(0, 1);  // x
    
    Polynomial u = x;
    FiniteField p = FiniteField::fromParts(0, 0x8000000000000000ULL);  // 模数p
    
    // 计算 x^p mod f
    for (int i = 0; i < 127; i++) {  // p的比特数
        u = (u * u) % (*this);
        if ((p.getLow() >> i) & 1) {
            u = (u * x) % (*this);
        }
    }
    
    // 检查 gcd(x^p - x, f)
    Polynomial diff = u - x;
    return gcd(diff).degree() == 0;
}
