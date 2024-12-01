#include <iostream>
#include <string>
#include <vector>
#include "cryptopp/cryptlib.h"
#include "cryptopp/shake.h"
#include "cryptopp/hex.h"
#include "cryptopp/pch.h"
#include "cryptopp/secblock.h"
#include <cryptopp/config.h>
#include <cryptopp/keccak.h>
#include "finite_field.h"
extern "C" {
    #include "KeccakHash.h"
}

using namespace std;
using namespace CryptoPP;

// 明确指定使用 unsigned char 替代 byte
using KeccakByte = unsigned char;

static void initShake(const FiniteField &p, const string &context, Keccak_HashInstance &shake) {
    // 更新"Hydra"字符串
    size_t bitlen = 128; // p的位长0x8000000000000000000000000000002d
    size_t num = (bitlen + 63) / 64; // p的块数
    if (Keccak_HashUpdate(&shake, (const KeccakByte*)"Hydra", 5 * 8) != KECCAK_SUCCESS) {
        printf("Failed to update Hydra\n");
    }
    
    // 更新context字符串
    if (Keccak_HashUpdate(&shake, (const KeccakByte*)context.data(), context.size() * 8) != KECCAK_SUCCESS) {
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

// 测试SHAKE128哈希函数
void testShake128() {
    cout << "测试 SHAKE128:" << endl;
    
    Keccak_HashInstance shakeMatrices;
    KeccakByte output[32]; // 添加输出缓冲区
    
    if (Keccak_HashInitialize_SHAKE128(&shakeMatrices) != KECCAK_SUCCESS) {
        printf("Failed to initialize shakeMatrices\n");
    }
    
    initShake(FiniteField::fromParts(0, FiniteField::getModule().low), "Matrices", shakeMatrices);
    
    // 使用 Keccak_HashSqueeze 来获取输出
    if (Keccak_HashSqueeze(&shakeMatrices, output, 32 * 8) != KECCAK_SUCCESS) {
        printf("Failed to squeeze hash output\n");
    }
    
    // 打印结果
    for (int i = 0; i < 32; i++) {
        printf("%02x", output[i]);
    }
    printf("\n");
}

int main() {
    try {
        testShake128();
        return 0;
    } catch(const Exception& e) {
        cerr << "错误: " << e.what() << endl;
        return 1;
    }
}
