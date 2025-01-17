/*
替换思路：
注意如果是有限域运算，需要手动调用mod()函数
使用FiniteField准确代替ZZ_p，近似代替mpz_class，因为mpz_class可以表示无穷大的数，这一点我们暂时实现不了，也用不上。
*/

#include "Hydra.h"


void print_outputshake(Keccak_HashInstance shake) {
     // 使用 Keccak_HashSqueeze 来获取输出
    KeccakByte output[32];
    if (Keccak_HashSqueeze(&shake, output, 32 * 8) != KECCAK_SUCCESS) {
        printf("Failed to squeeze hash output\n");
    }

    // 打印结果
    for (int i = 0; i < 32; i++) {
        printf("%02x", output[i]);
    }
    printf("\n");
}

// 构造函数
__host__ Hydra::Hydra(FiniteField p, int t, int kappa) 
    : p(p), kappa(kappa) {
    
    this->F = FiniteField::fromParts(0, 1);
    this->kappa = kappa;
    this->perms = num_perms(t);
    this->d = get_d(p);
    Rounds rounds = get_rounds(p, kappa, d);
    this->Re_1 = rounds.Re_1;
    this->Re_2 = rounds.Re_2;
    this->Ri = rounds.Ri;
    this->Rh = rounds.Rh;

    // 初始化矩阵 Me
    Me = Matrix(4, 4);
    Me.at(0, 0) = FiniteField::fromParts(0, 3);
    Me.at(0, 1) = FiniteField::fromParts(0, 2);
    Me.at(0, 2) = FiniteField::fromParts(0, 1);
    Me.at(0, 3) = FiniteField::fromParts(0, 1);
    Me.at(1, 0) = FiniteField::fromParts(0, 1);
    Me.at(1, 1) = FiniteField::fromParts(0, 3);
    Me.at(1, 2) = FiniteField::fromParts(0, 2);
    Me.at(1, 3) = FiniteField::fromParts(0, 1);
    Me.at(2, 0) = FiniteField::fromParts(0, 1);
    Me.at(2, 1) = FiniteField::fromParts(0, 1);
    Me.at(2, 2) = FiniteField::fromParts(0, 3);
    Me.at(2, 3) = FiniteField::fromParts(0, 2);
    Me.at(3, 0) = FiniteField::fromParts(0, 2);
    Me.at(3, 1) = FiniteField::fromParts(0, 1);
    Me.at(3, 2) = FiniteField::fromParts(0, 1);
    Me.at(3, 3) = FiniteField::fromParts(0, 3);

    // 初始化 SHAKE128 并生成矩阵
    Keccak_HashInstance shakeMatrices;
    if (Keccak_HashInitialize_SHAKE128(&shakeMatrices) != KECCAK_SUCCESS) {
        printf("Failed to initialize shakeMatrices\n");
    }
    initShake(FiniteField::getFiniteFieldModule(), "Matrices", shakeMatrices);
    // print_outputshake(shakeMatrices);
    Mi = gen_mi(shakeMatrices);
    Mh = gen_mh(shakeMatrices);
    // 这之前没啥问题
    // 生成常量
    Keccak_HashInstance shakeConstants;
    if (Keccak_HashInitialize_SHAKE128(&shakeConstants) != KECCAK_SUCCESS) {
        printf("Failed to initialize shakeConstants\n");
    }
    initShake(FiniteField::getFiniteFieldModule(), "Constants", shakeConstants);

    rc_b = gen_rc(Re_1 + Re_2 + Ri, 4, shakeConstants);
    rc_h = gen_rc(Rh, 8, shakeConstants);
    // 初始化 Rolling
    Keccak_HashInstance shakeRolling;
    if (Keccak_HashInitialize_SHAKE128(&shakeRolling) != KECCAK_SUCCESS) {
        printf("Failed to initialize shakeRolling\n");
    }
    initShake(FiniteField::getFiniteFieldModule(), "Rolling", shakeRolling);
    rc_r = gen_rc(perms - 1, 8, shakeRolling);
}

// SHAKE128 相关函数
__host__ void Hydra::initShake(const FiniteField &p, const char* context, Keccak_HashInstance &shake) {
    // 更新"Hydra"字符串
    size_t bitlen = 128; // p的位长0x8000000000000000000000000000002d
    size_t num = (bitlen + 63) / 64; // p的块数
    if (Keccak_HashUpdate(&shake, (const KeccakByte*)"Hydra", 5 * 8) != KECCAK_SUCCESS) {
        printf("Failed to update Hydra\n");
    }
    
    // 更新context字符串
    size_t context_len = c_strlen(context);
    if (Keccak_HashUpdate(&shake, (const KeccakByte*)context, context_len * 8) != KECCAK_SUCCESS) {
        printf("Failed to update context\n");
    }

    // 只需要获取p的低128位，因为p的位长是128位
    uint128_t tmp = p.getLow();
    for (size_t i = 0; i < num; i++) {
        uint64_t prime_block = tmp & 0xFFFFFFFFFFFFFFFF;
        KeccakByte block[8];
        for (int j = 0; j < 8; j++) {
            block[j] = (prime_block >> (j * 8)) & 0xFF;
        }
        if (Keccak_HashUpdate(&shake, block, 8 * 8) != KECCAK_SUCCESS) {
            printf("Failed to update context\n");
        }
        tmp >>= 64;
    }
    if (Keccak_HashFinal(&shake, NULL) != KECCAK_SUCCESS) {
        printf("Failed to finalize the input data\n");
    }
}

__host__ FiniteField Hydra::field_element_from_shake(Keccak_HashInstance &shake) {
    size_t bitlen = 128; // p的位长0x8000000000000000000000000000002d 128位
    size_t byte = (bitlen + 7) / 8;
    size_t word = (byte + 7) / 8;
    int k = 0;
    while (true) {
        uint64_t word_buf[2];
        for (size_t j = 0; j < 2; j++) {
            word_buf[j] = 0;
        }
        ByteBuffer buf(byte);
        if (Keccak_HashSqueeze(&shake, buf.data, buf.size * 8) != KECCAK_SUCCESS) {
            printf("Failed to squeeze the buf\n");
        }
        for (size_t i = 0; i < word; i++) {
            uint64_t value = 0;
            for (size_t j = i * 8; j < min((i + 1) * 8, byte); j++) {
                value |= (uint64_t)buf[j] << ((j - i * 8) * 8);
            }
            word_buf[i] = value;
        }
        // res是mpz_class类型，不能取模
        FiniteField res = FiniteField::fromParts(0, 0);
        for (int i = 1; i >= 0; i--) {
            res = (res << 64) | FiniteField::fromParts(0, word_buf[i]);  // 使用位或运算符 | 代替加法
        }
        // res.printToStream(std::cout);
        // printf("\n");
        if (res < p) {
            return res;
        }
        k += 1;
    }
}

 __host__ FiniteField Hydra::field_element_from_shake_without_0(Keccak_HashInstance &shake) {
    while (true) {
        FiniteField el = field_element_from_shake(shake);
        if (el != FiniteField::fromParts(0, 0)) {
            return el;
        }
    }
}

// 辅助函数实现
 __host__ __device__ int Hydra::get_R_star(int kappa) {
    assert(kappa >= 80 && kappa <= 256);
    int R_star[177] = {
    19, 19, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 22, 22, 23,
    23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 26,
    26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 28, 28, 29, 29, 29, 29, 29,
    30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32,
    33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 35, 35, 36, 36, 36, 36,
    36, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39,
    39, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42,
    43, 43, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 46, 46, 46, 46,
    46, 46, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49,
    49, 50, 50, 50, 50, 50, 50, 51, 51, 52, 52, 52, 52, 52, 53, 53, 53,
    53, 53, 53, 54, 54, 54, 54
    };
    return R_star[kappa - 80];
}

 __host__ __device__ int Hydra::num_perms(int t) {
    int t_ = t / 8;
    int t__ = t % 8;
    int perms = t_;
    if (t__ > 0) {
        perms += 1;
    }
    return perms;
}

 __host__ __device__ double log2_finitefield(const FiniteField& x) {
    // 找到最高位的1的位置
    int msb = -1;
    uint128_t value = x.getLow();
    while (value != 0) {
        value >>= 1;
        msb++;
    }
    
    return (double)msb;  // 返回最高位的位置，即约等于log2(x)
}


 __host__ __device__ int Hydra::get_round_num_head(FiniteField p, int kappa) {
    int R_star = get_R_star(kappa);
    double x0 = kappa / 24.0 + log2(12);
    double x1 = (kappa - log2_finitefield(p)) / 22.0 + log2(11);
    int R_hat = 3 + ceil(max_double(x0, x1));
    int R = ceil(1.25 * ceil(max_three_double(24, R_hat, R_star + 2)));
    return R;
}

 __host__  int Hydra::get_round_num_internal(FiniteField p, int kappa, int d) {
    double x0 = kappa / 16.0;
    double x1 = (kappa - log2_finitefield(p)) / 12.0;  // 替换原来的 mpz 版本
    int R_hat = 4 - floor(log2(d)) + ceil(max_double(x0, x1));
    int R = ceil(1.125 * ceil(max_double(kappa / 4.0 - log2(static_cast<double>(d)) + 6, static_cast<double>(R_hat))));
    return R;
}

 __host__ __device__ Rounds Hydra::get_rounds(FiniteField p, int kappa, int d) {
    int Re_1 = 2;
    int Re_2 = 4;
    int Ri = get_round_num_internal(p, kappa, d);
    int Rh = get_round_num_head(p, kappa);
    return Rounds(Re_1, Re_2, Ri, Rh);
}

// 首先需要一个辅助函数来计算最大公约数 (GCD)
 __host__ __device__ FiniteField GCD(FiniteField a, FiniteField b) {
    while (!(b.getLow() == 0)) {
        FiniteField temp = b;
        // b = a % b
        b = FiniteField::fromParts(0, a.getLow() % b.getLow());
        // a = temp
        a = temp;
    }
    return a;
}

 __host__ __device__ int Hydra::get_d(const FiniteField& p) {
    for (int d = 3; d < p.getLow(); ++d) {
        if (GCD(FiniteField::fromParts(0, d), p - FiniteField::fromParts(0, 1)).getLow() == 1) {
            return d;
        }
    }
    return -1;
}

// 矩阵生成函数
__host__ Matrix Hydra::gen_mi(Keccak_HashInstance &shake) {
    FiniteFieldArray lambda1(4);
    lambda1[0] = FiniteField::fromParts(0, 1);
    lambda1[1] = FiniteField::getFiniteFieldModule() - FiniteField::fromParts(0, 1); // -1 = p - 1
    lambda1[2] = FiniteField::fromParts(0, 1);
    lambda1[3] = FiniteField::getFiniteFieldModule() - FiniteField::fromParts(0, 1);

    FiniteFieldArray lambda2(4);
    lambda2[0] = FiniteField::fromParts(0, 1);
    lambda2[1] = FiniteField::fromParts(0, 1);
    lambda2[2] = FiniteField::getFiniteFieldModule() - FiniteField::fromParts(0, 1);
    lambda2[3] = FiniteField::getFiniteFieldModule() - FiniteField::fromParts(0, 1);
    while (true) {
        Matrix M = gen_matrix(shake, 4);
        if (M.check_conditions(lambda1, 4) && M.check_conditions(lambda2, 4)) {
            return M;
        }
    }
}

 __host__ Matrix Hydra::gen_matrix(Keccak_HashInstance &shake, int size) {
    Matrix M(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            M.at(i, j) = FiniteField::fromParts(0, 1);
        }
    }
    M.at(0, 0) = field_element_from_shake_without_0(shake);
    for (int i = 1; i < size; ++i) {
        M.at(i, 0) = field_element_from_shake_without_0(shake);
        M.at(i, i) = field_element_from_shake_without_0(shake);
    }
    return M;
}

 __host__ Matrix Hydra::gen_mh(Keccak_HashInstance &shake) {
    Matrix result(8, 8);
    FiniteFieldArray lambdas(8);
    lambdas[0] = FiniteField::fromParts(0, 1);
    lambdas[1] = FiniteField::fromParts(0, 1);
    lambdas[2] = FiniteField::fromParts(0, 1);
    lambdas[3] = FiniteField::fromParts(0, 1);
    lambdas[4] = FiniteField::getFiniteFieldModule() - FiniteField::fromParts(0, 1);
    lambdas[5] = FiniteField::getFiniteFieldModule() - FiniteField::fromParts(0, 1);
    lambdas[6] = FiniteField::getFiniteFieldModule() - FiniteField::fromParts(0, 1);
    lambdas[7] = FiniteField::getFiniteFieldModule() - FiniteField::fromParts(0, 1);
    while (true) {
        Matrix M = gen_matrix(shake, 8);
        if (M.check_conditions(lambdas, 8)) {
            return M;
        }
    }
}

 __host__ FiniteFieldArray* Hydra::gen_rc(int rounds, int size, Keccak_HashInstance &shake) {
    // 检查参数有效性
    if (rounds <= 0 || size <= 0) {
        return nullptr;
    }
    
    // 创建数组
    FiniteFieldArray* rc = new FiniteFieldArray[rounds];
    
    // 初始化每个FiniteFieldArray
    for (int i = 0; i < rounds; ++i) {
        // 直接在构造时指定大小
        rc[i] = FiniteFieldArray(size);
        
        // 生成并填充随机元素
        for (int j = 0; j < size; ++j) {
            FiniteField value = field_element_from_shake(shake);
            rc[i][j] = value;
        }
    }
    
    return rc;
}

 __host__ __device__ DotPair Hydra::get_lm_dot(const FiniteFieldArray& state) {
    // 确保输入数组长度为4
    #ifndef __CUDA_ARCH__
    if (state.getSize() != 4) {
        throw std::runtime_error("State array must have length 4");
    }
    #endif
    
    // 计算 tmp = state[0] - state[3]
    FiniteField tmp = state[0] - state[3];
    
    // 计算 dot1 = tmp - state[1] + state[2]
    FiniteField dot1 = (tmp - state[1] + state[2]).mod();
    
    // 计算 dot2 = tmp + state[1] - state[2]
    FiniteField dot2 = (tmp + state[1] - state[2]).mod();
    
    return DotPair(dot1, dot2);
}

// ��线性变换函数
__host__ __device__ FiniteFieldArray Hydra::non_linear_e(const FiniteFieldArray& state) {
    FiniteFieldArray result(state.getSize());
    for (int i = 0; i < state.getSize(); ++i) {
        result[i] = FiniteField::power(state[i], d).mod();
    }
    return result;
}

__host__ __device__ FiniteFieldArray Hydra::non_linear_i(const FiniteFieldArray& state) {
    DotPair dot_pair = get_lm_dot(state);
    FiniteField dot1 = dot_pair.dot1;
    FiniteField dot2 = dot_pair.dot2;
    dot1 = (dot1 * dot1).mod();
    FiniteField sum_ = (dot1 + dot2).mod();
    FiniteField prod = FiniteField::power(sum_, 2).mod();
    FiniteFieldArray result(state.getSize());
    for (int i = 0; i < state.getSize(); ++i) {
        result[i] = (state[i] + prod).mod();
    }
    return result;
}


__host__ __device__ FiniteFieldArray Hydra::non_linear_h(const FiniteFieldArray& state) {
    // assert(state.getSize() == 8);
    FiniteField dot = state[0] + state[1] + state[2] + state[3] - state[4] - state[5] - state[6] - state[7];
    dot = (dot * dot).mod();
    FiniteFieldArray result(state.getSize());
    for (int i = 0; i < state.getSize(); ++i) {
        result[i] = (state[i] + dot).mod();
    }
    return result;
}

__device__ DotPair Hydra::non_linear_r(FiniteFieldArray y, FiniteFieldArray z) {
    DotPair dot_pair_y = get_lm_dot(y);
    DotPair dot_pair_z = get_lm_dot(z);
    FiniteField vy = dot_pair_y.dot1;
    FiniteField wy = dot_pair_y.dot2;
    FiniteField wz = dot_pair_z.dot2;
    FiniteField vz = dot_pair_z.dot1;
    FiniteField v = (vy * vz).mod();
    FiniteField w = (wy * wz).mod();
    for (int i = 0; i < y.getSize(); ++i) {
        y[i] = (y[i] + v).mod();
        z[i] = (z[i] + w).mod();
    }
    return DotPair(v, w);
}

__device__ FiniteFieldArray Hydra::R(const FiniteFieldArray& state, int i) {
    // assert(state.getSize() == 8);
    // assert(rc_r.size() >= i);
    if (i == 0) {
        // 从这里返回
        return state;
    }
    FiniteFieldArray y = slice(state, 0, 4);
    FiniteFieldArray z = slice(state, 4, 8);
    DotPair dot_pair = non_linear_r(y, z);
    // [4,4] * [4,1] = [4,1]
    FiniteFieldArray y_perm = (Mi * y).mod();
    FiniteFieldArray z_perm = (Mi * z).mod();
    FiniteFieldArray result = (concat_vec(y_perm, z_perm) + rc_r[i - 1]).mod();
    return result;
}

 __host__ __device__ FiniteFieldArray Hydra::concat(const FiniteFieldArray& a, const FiniteFieldArray& b) {
    FiniteFieldArray result(a.getSize() + b.getSize());
    for (int i = 0; i < a.getSize(); ++i) {
        result[i] = a[i];
    }
    for (int i = 0; i < b.getSize(); ++i) {
        result[a.getSize() + i] = b[i];
    }
    return result;
}

 __host__ FiniteFieldArray Hydra::concat_vec(const FiniteFieldArray& a, const FiniteFieldArray& b) {
    size_t size_a = a.getSize();
    size_t size_b = b.getSize();
    FiniteFieldArray result(size_a + size_b);
    
    // 复制第一个数组的元素
    for (size_t i = 0; i < size_a; ++i) {
        result[i] = a[i];
    }
    
    // 复制第二个数组的元素
    for (size_t i = 0; i < size_b; ++i) {
        result[size_a + i] = b[i];
    }
    
    return result;
}

// 置换函数
__host__ __device__ StateSumPair Hydra::permutation_b(FiniteFieldArray state) {
    FiniteFieldArray sum_(4); 
    // 这个循环出错
    for (int i = 0; i < 4; ++i) {
        sum_[i] = FiniteField::fromParts(0, 0);  // 初始化为0
    }

    state = Me * state;
    for (int i = 0; i < Re_1; ++i) {
        state = non_linear_e(state);
        state = Me * state + rc_b[i];
        sum_ = sum_ + state;
    }
    for (int i = 0; i < Ri; ++i) {
        state = non_linear_i(state);
        state = Mi * state + rc_b[i + Re_1];
        sum_ = sum_ + state;
    }
    for (int i = Re_1; i < Re_1 + Re_2; ++i) {
        state = non_linear_e(state);
        state = Me * state + rc_b[i + Ri];
        if (i < Re_1 + Re_2 - 1) {
            sum_ = sum_ + state;
        }
    }
    return StateSumPair(state, sum_);
}

__host__ __device__ FiniteFieldArray Hydra::permutation_h(FiniteFieldArray state, const FiniteFieldArray& K) {
    for (int i = 0; i < Rh; ++i) {
        state = non_linear_h(state);
        state = (Mh * state + rc_h[i]).mod();
        state = (state + K).mod();
    }
    return state;
}

__host__ __device__ FiniteFieldArray Hydra::gen_ks(int t, const FiniteFieldArray& K, const FiniteFieldArray& IV, const FiniteFieldArray& N) {
    assert(IV.getSize() == 3);
    assert(K.getSize() == 4);
    
    FiniteFieldArray state = concat(N, IV);
    assert(state.getSize() == 4);
    
    FiniteFieldArray K_vec = K;
    state = state + K_vec;
    
    StateSumPair result = permutation_b(state); 
    state = result.state;
    FiniteFieldArray z = result.sum;
    
    state = state + K_vec;
    FiniteFieldArray K_mat = Me * K_vec;
    FiniteFieldArray K_ = concat_vec(K_vec, K_mat);
    
    FiniteFieldArray keystream(t);
    int perm_counter = -1;
    FiniteFieldArray roll = concat_vec(state, z);
    FiniteFieldArray perm;
    
    for (int i = 0; i < t; ++i) {
        int off = i % 8;
        if (off == 0) {
            perm_counter++;
            roll = R(roll, perm_counter);
            perm = permutation_h(roll, K_);
            perm = perm + roll; 
        }
        keystream[i] = perm[off];
    }

    return keystream;
}

// 加密解密函数
// __device__ FiniteFieldArray Hydra::encrypt(
//     const FiniteFieldArray& plains,
//     const FiniteFieldArray& K,
//     const FiniteFieldArray& IV,
//     const FiniteFieldArray& N
// ) {
//     FiniteFieldArray keystream = gen_ks(plains.getSize(), K, IV, N);
//     FiniteFieldArray ciphers = plains + keystream;
//     return ciphers;
// }

__device__ FiniteFieldArray Hydra::decrypt(
    const FiniteFieldArray& ciphers,
    const FiniteFieldArray& K,
    const FiniteFieldArray& IV,
    const FiniteFieldArray& N
) {
    FiniteFieldArray keystream = gen_ks(ciphers.getSize(), K, IV, N);
    FiniteFieldArray plains = ciphers - keystream;
    return plains;
}

__host__ __device__ FiniteFieldArray Hydra::slice(const FiniteFieldArray& state, int start, int end) {
    // assert(start >= 0 && end <= state.getSize());
    FiniteFieldArray result(end - start);
    for (int i = start; i < end; ++i) {
        result[i - start] = state[i];
    }
    return result;
}

__host__ __device__ double Hydra::max_double(double a, double b) {
    return a > b ? a : b;
}

__host__ __device__ double Hydra::max_three_double(double a, double b, double c) {
    return max_double(max_double(a, b), c);
}

__host__ Hydra* Hydra::copyToDevice() {
    // 1. 为 Hydra 对象分配设备端内存
    Hydra* d_hydra;
    cudaMalloc(&d_hydra, sizeof(Hydra));

    // 2. 将 Hydra 的基本结构复制到设备端
    cudaMemcpy(d_hydra, this, sizeof(Hydra), cudaMemcpyHostToDevice);

    // 3. 复制 rc_b 数组到设备端
    if (rc_b) {
        int rc_b_size = Re_1 + Re_2 + Ri; // 计算总大小
        FiniteFieldArray* d_rc_b;
        cudaMalloc(&d_rc_b, sizeof(FiniteFieldArray) * rc_b_size);

        for (int i = 0; i < rc_b_size; ++i) {
            FiniteFieldArray temp_array = rc_b[i];

            // 为设备端的 elements 指针分配内存
            FiniteField* d_elements;
            cudaMalloc(&d_elements, sizeof(FiniteField) * temp_array.getSize());
            
            // 复制 elements 的内容到设备端
            cudaMemcpy(d_elements, temp_array.getElements(), sizeof(FiniteField) * temp_array.getSize(), cudaMemcpyHostToDevice);

            // 在主机端更新 temp_array 的 elements 指针为设备地址
            temp_array.setElements(d_elements);

            // 将更新后的 temp_array 复制到设备端
            cudaMemcpy(&d_rc_b[i], &temp_array, sizeof(FiniteFieldArray), cudaMemcpyHostToDevice);
        }

        // 将 d_rc_b 的指针赋值给设备端 Hydra 对象
        cudaMemcpy(&(d_hydra->rc_b), &d_rc_b, sizeof(FiniteFieldArray*), cudaMemcpyHostToDevice);
    }


    // 4. 复制 rc_h 数组到设备端
    if (rc_h) {
        int rc_h_size = Rh; // 计算总大小
        FiniteFieldArray* d_rc_h;
        cudaMalloc(&d_rc_h, sizeof(FiniteFieldArray) * rc_h_size);

        for (int i = 0; i < rc_h_size; ++i) {
            FiniteFieldArray temp_array = rc_h[i];

            // 为设备端的 elements 指针分配内存
            FiniteField* d_elements;
            cudaMalloc(&d_elements, sizeof(FiniteField) * temp_array.getSize());
            
            // 复制 elements 的内容到设备端
            cudaMemcpy(d_elements, temp_array.getElements(), sizeof(FiniteField) * temp_array.getSize(), cudaMemcpyHostToDevice);

            // 在主机端更新 temp_array 的 elements 指针为设备地址
            temp_array.setElements(d_elements);

            // 将更新后的 temp_array 复制到设备端
            cudaMemcpy(&d_rc_h[i], &temp_array, sizeof(FiniteFieldArray), cudaMemcpyHostToDevice);
        }

        // 将 d_rc_b 的指针赋值给设备端 Hydra 对象
        cudaMemcpy(&(d_hydra->rc_h), &d_rc_h, sizeof(FiniteFieldArray*), cudaMemcpyHostToDevice);
    }

    // 5. 复制 rc_r 数组到设备端
    if (rc_r) {
        int rc_r_size = perms; // 计算总大小
        FiniteFieldArray* d_rc_r;
        cudaMalloc(&d_rc_r, sizeof(FiniteFieldArray) * rc_r_size);

        for (int i = 0; i < rc_r_size; ++i) {
            FiniteFieldArray temp_array = rc_r[i];

            // 为设备端的 elements 指针分配内存
            FiniteField* d_elements;
            cudaMalloc(&d_elements, sizeof(FiniteField) * temp_array.getSize());
            
            // 复制 elements 的内容到设备端
            cudaMemcpy(d_elements, temp_array.getElements(), sizeof(FiniteField) * temp_array.getSize(), cudaMemcpyHostToDevice);

            // 在主机端更新 temp_array 的 elements 指针为设备地址
            temp_array.setElements(d_elements);

            // 将更新后的 temp_array 复制到设备端
            cudaMemcpy(&d_rc_r[i], &temp_array, sizeof(FiniteFieldArray), cudaMemcpyHostToDevice);
        }

        // 将 d_rc_b 的指针赋值给设备端 Hydra 对象
        cudaMemcpy(&(d_hydra->rc_r), &d_rc_r, sizeof(FiniteFieldArray*), cudaMemcpyHostToDevice);
    }

    return d_hydra;
}


__host__ void Hydra::freeDeviceCopy(Hydra* d_hydra) {
    Hydra h_temp;
    // 从设备端复制数据到主机端，获取设备端指针
    cudaMemcpy(&h_temp, d_hydra, sizeof(Hydra), cudaMemcpyDeviceToHost);

    // 释放设备端 rc_b
    if (h_temp.rc_b) {
        int rc_b_size = h_temp.Re_1 + h_temp.Re_2 + h_temp.Ri;
        for (int i = 0; i < rc_b_size; ++i) {
            cudaFree(&h_temp.rc_b[i]);
        }
        cudaFree(h_temp.rc_b);
    }

    // 释放设备端 rc_h
    if (h_temp.rc_h) {
        int rc_h_size = h_temp.Rh;
        for (int i = 0; i < rc_h_size; ++i) {
            cudaFree(&h_temp.rc_h[i]);
        }
        cudaFree(h_temp.rc_h);
    }

    // 释放设备端 rc_r
    if (h_temp.rc_r) {
        int rc_r_size = h_temp.perms;
        for (int i = 0; i < rc_r_size; ++i) {
            cudaFree(&h_temp.rc_r[i]);
        }
        cudaFree(h_temp.rc_r);
    }

    // 释放设备端 Hydra 对象本身
    cudaFree(d_hydra);
}

// __device__ void applyHydra(Hydra* hydra, FiniteField* ct1, FiniteField* ct2, FiniteField* ct3, FiniteField* ct4) {
//     // 初始化明文数据
//     FiniteFieldArray plains(4);
//     plains[0] = FiniteField::fromParts(0, 0); // 明文块 0
//     plains[1] = FiniteField::fromParts(0, 1); // 明文块 1
//     plains[2] = FiniteField::fromParts(0, 2); // 明文块 2
//     plains[3] = FiniteField::fromParts(0, 3); // 明文块 3

//     // 初始化 K, IV, N
//     FiniteFieldArray K(4), IV(3), N(1);
//     for (int i = 0; i < 4; ++i) {
//         K[i] = FiniteField::fromParts(0, 0);  // 密钥初始化为0
//     }
//     for (int i = 0; i < 3; ++i) {
//         IV[i] = FiniteField::fromParts(0, 1); // 初始化向量为1
//     }
//     N[0] = FiniteField::fromParts(0, 2);      // 随机数为2

//     // 调用 Hydra 加密
//     FiniteFieldArray encrypted = hydra->encrypt(plains, K, IV, N);

//     // 将加密结果写入输出参数
//     *ct1 = encrypted[0];
//     *ct2 = encrypted[1];
//     *ct3 = encrypted[2];
//     *ct4 = encrypted[3];
//     printf("encrypted[0]: ");
//     encrypted[0].print();
//     printf("\n");
//     printf("encrypted[1]: ");
//     encrypted[1].print();
//     printf("\n");
//     printf("encrypted[2]: ");
//     encrypted[2].print();
//     printf("\n");
//     printf("encrypted[3]: ");
//     encrypted[3].print();
//     printf("\n");
// }

// __global__ void hydraEncrypt(FiniteFieldArray* d_state_out, Hydra* d_hydra, int t) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (idx < t / 4) {
//         FiniteField ct1, ct2, ct3, ct4;
//         applyHydra(d_hydra, &ct1, &ct2, &ct3, &ct4);

//         // 现在可以正常使用 setElement 方法
//         d_state_out->setElement(idx * 4, ct1);
//         d_state_out->setElement(idx * 4 + 1, ct2);
//         d_state_out->setElement(idx * 4 + 2, ct3);
//         d_state_out->setElement(idx * 4 + 3, ct4);
//     }
// }


