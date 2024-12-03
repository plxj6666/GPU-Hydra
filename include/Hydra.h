#ifndef HYDRA_H
#define HYDRA_H

#include "finite_field.h"
#include "matrix.h"
#include "polynomial.h"
#include <cryptopp/shake.h>
#include <cstring>
#include <cmath>
using KeccakByte = unsigned char;

// 前向声明
extern "C" {
    #include "KeccakHash.h"
}

// 用于get_rounds返回的结构体
struct Rounds {
    int Re_1;
    int Re_2;
    int Ri;
    int Rh;
    
    __host__ __device__ Rounds(int re1, int re2, int ri, int rh) 
        : Re_1(re1), Re_2(re2), Ri(ri), Rh(rh) {}
};

// 用于get_lm_dot返回的结构体
struct DotPair {
    FiniteField dot1;
    FiniteField dot2;
    
    __host__ __device__ DotPair(FiniteField d1, FiniteField d2) 
        : dot1(d1), dot2(d2) {}
};

// 用于permutation_b返回的结构体
struct StateSumPair {
    FiniteFieldArray state;
    FiniteFieldArray sum;
    
    __host__ __device__ StateSumPair(const FiniteFieldArray& s, const FiniteFieldArray& sum_)
        : state(s), sum(sum_) {}
};

// 用于替代 SecByteBlock 的简单缓冲区
struct ByteBuffer {
    unsigned char* data;
    size_t size;
    
    __host__ ByteBuffer(size_t s) : size(s) {
        data = new unsigned char[s];
    }
    
    __host__ ~ByteBuffer() {
        if (data) delete[] data;
    }

    __host__ unsigned char& operator[](size_t index) {
        assert(index < size);
        return data[index];
    }
};

class Hydra {
private:
    // 基础参数
    FiniteField p;  // 有限域模数
    FiniteField F;
    int kappa;
    int perms;
    int d;
    int Re_1, Re_2, Ri, Rh;
    
    // 矩阵
    Matrix Me, Mi, Mh;
    
    // 常量数组
    FiniteFieldArray* rc_b;
    FiniteFieldArray* rc_h;
    FiniteFieldArray* rc_r;

    // 私有辅助函数
    __host__ __device__ static int get_R_star(int kappa);
    __host__ __device__ static int get_round_num_head(FiniteField p, int kappa);
    __host__ __device__ static int get_round_num_internal(FiniteField p, int kappa, int d);
    __host__ __device__ static int get_d(const FiniteField& p);
    __host__ __device__ static Rounds get_rounds(FiniteField p, int kappa, int d);
    __host__ __device__ static int num_perms(int t);
    // SHAKE128 相关函数
    __host__ void initShake(const FiniteField &p, const char* context, Keccak_HashInstance &shake);
    __host__ FiniteField field_element_from_shake(Keccak_HashInstance &shake);
    __host__ FiniteField field_element_from_shake_without_0(Keccak_HashInstance &shake);
    
    // 矩阵生成函数
    __host__ Matrix gen_matrix(Keccak_HashInstance &shake, int size);
    __host__ Matrix gen_mi(Keccak_HashInstance &shake);
    __host__ Matrix gen_mh(Keccak_HashInstance &shake);
    __host__ FiniteFieldArray* gen_rc(int rounds, int size, Keccak_HashInstance &shake);

    // 非线性变换函数
    __host__ __device__ FiniteFieldArray non_linear_e(const FiniteFieldArray& state);
    __host__ __device__ FiniteFieldArray non_linear_i(const FiniteFieldArray& state);
    __host__ __device__ FiniteFieldArray non_linear_h(const FiniteFieldArray& state);
    __host__ __device__ DotPair non_linear_r(FiniteFieldArray y, FiniteFieldArray z);
    
    // 辅助函数
    __host__ __device__ DotPair get_lm_dot(const FiniteFieldArray& state);
    __host__ __device__ static FiniteFieldArray concat(const FiniteFieldArray& a, const FiniteFieldArray& b);
    __host__ __device__ static FiniteFieldArray concat_vec(const FiniteFieldArray& a, const FiniteFieldArray& b);
    __host__ __device__ static FiniteFieldArray slice(const FiniteFieldArray& state, int start, int end);
    __host__ __device__ static double max_double(double a, double b);
    __host__ __device__ static double max_three_double(double a, double b, double c);
    
    // 置换函数
    __host__ __device__ StateSumPair permutation_b(FiniteFieldArray state);
    __host__ __device__ FiniteFieldArray permutation_i(FiniteFieldArray state);
    __host__ __device__ FiniteFieldArray permutation_h(FiniteFieldArray state, const FiniteFieldArray& K);
    __host__ __device__ FiniteFieldArray R(const FiniteFieldArray& state, int i);

    // 辅助函数 - 添加字符串长度计算函数
    __host__ __device__ static size_t c_strlen(const char* str) {
        size_t len = 0;
        while(str[len] != '\0') len++;
        return len;
    }

    bool device_pointers;  // 新增成员

public:
    // 构造函数和析构函数
    __host__ Hydra(FiniteField p, int t, int kappa);
    __host__ ~Hydra() {
        if (!device_pointers) {  // 只有当不是设备指针时才删除
            if (rc_b) delete[] rc_b;
            if (rc_h) delete[] rc_h;
        }
    }
    
    // 密钥生成和加密解密函数
    __host__ __device__ FiniteFieldArray gen_ks(int t, const FiniteFieldArray& K, const FiniteFieldArray& IV, const FiniteFieldArray& N);
    __host__ __device__ FiniteFieldArray encrypt(
        const FiniteFieldArray& plains,
        const FiniteFieldArray& K,
        const FiniteFieldArray& IV,
        const FiniteFieldArray& N
    );
    
    __device__ FiniteFieldArray decrypt(
        const FiniteFieldArray& ciphers,
        const FiniteFieldArray& K,
        const FiniteFieldArray& IV,
        const FiniteFieldArray& N
    );

    // 辅助函数
    __host__ __device__ FiniteField get_f() const { return F; }

    // 添加访问器方法
    __host__ __device__ Matrix& getME() { return Me; }
    __host__ __device__ Matrix& getMI() { return Mi; }
    __host__ __device__ Matrix& getMH() { return Mh; }
    __host__ __device__ FiniteFieldArray* getRCB() { return rc_b; }
    __host__ __device__ FiniteFieldArray* getRCH() { return rc_h; }
    __host__ __device__ int getRE1() const { return Re_1; }
    __host__ __device__ int getRE2() const { return Re_2; }
    __host__ __device__ int getRI() const { return Ri; }
    __host__ __device__ int getRH() const { return Rh; }
    
    // 设置器方法
    __host__ __device__ void setME(const Matrix& m) { Me = m; }
    __host__ __device__ void setMI(const Matrix& m) { Mi = m; }
    __host__ __device__ void setMH(const Matrix& m) { Mh = m; }
    __host__ __device__ void setRCB(FiniteFieldArray* rc) { rc_b = rc; }
    __host__ __device__ void setRCH(FiniteFieldArray* rc) { rc_h = rc; }
    __host__ __device__ void setDevicePointers(bool is_device) { device_pointers = is_device; }
};
__global__ void hydraEncrypt(FiniteFieldArray* output, Hydra* hydra, int t);
#endif // HYDRA_H
