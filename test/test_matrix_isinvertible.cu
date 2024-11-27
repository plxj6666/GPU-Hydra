#include "matrix.h"
#include <iostream>
#include <vector>
#include <random>

void testMatrixInvertibility() {
    std::cout << "\n=== Testing 8x8 Matrix Determinant and Invertibility ===\n" << std::endl;
    
    const int SIZE = 8;
    Matrix A(SIZE, SIZE);
    
    // 第一行
    A.at(0,0) = FiniteField::fromParts(0, 
        (uint128_t(0x71c68fb6a273b17aULL) << 64) | uint128_t(0x77a500f68c516b72ULL));
    A.at(0,1) = FiniteField::fromParts(0, 1);
    A.at(0,2) = FiniteField::fromParts(0, 1);
    A.at(0,3) = FiniteField::fromParts(0, 1);
    A.at(0,4) = FiniteField::fromParts(0, 1);
    A.at(0,5) = FiniteField::fromParts(0, 1);
    A.at(0,6) = FiniteField::fromParts(0, 1);
    A.at(0,7) = FiniteField::fromParts(0, 1);

    // 第二行
    A.at(1,0) = FiniteField::fromParts(0,
        (uint128_t(0x18e4f6cdb29e1ba6ULL) << 64) | uint128_t(0x9fbd1d153ac93e84ULL));
    A.at(1,1) = FiniteField::fromParts(0,
        (uint128_t(0x63809ffc2c18e13aULL) << 64) | uint128_t(0x6baea140d464d406ULL));
    A.at(1,2) = FiniteField::fromParts(0, 1);
    A.at(1,3) = FiniteField::fromParts(0, 1);
    A.at(1,4) = FiniteField::fromParts(0, 1);
    A.at(1,5) = FiniteField::fromParts(0, 1);
    A.at(1,6) = FiniteField::fromParts(0, 1);
    A.at(1,7) = FiniteField::fromParts(0, 1);

    // 第三行
    A.at(2,0) = FiniteField::fromParts(0,
        (uint128_t(0x0a52ab9853e2aa10ULL) << 64) | uint128_t(0x3101c54cb71ed78eULL));
    A.at(2,1) = FiniteField::fromParts(0, 1);
    A.at(2,2) = FiniteField::fromParts(0,
        (uint128_t(0x2cbf68db3916868aULL) << 64) | uint128_t(0x68e77035c9a2f2bfULL));
    A.at(2,3) = FiniteField::fromParts(0, 1);
    A.at(2,4) = FiniteField::fromParts(0, 1);
    A.at(2,5) = FiniteField::fromParts(0, 1);
    A.at(2,6) = FiniteField::fromParts(0, 1);
    A.at(2,7) = FiniteField::fromParts(0, 1);

    // 第四行
    A.at(3,0) = FiniteField::fromParts(0,
        (uint128_t(0x2b6c9add06474801ULL) << 64) | uint128_t(0xd2e4ed42603d67f1ULL));
    A.at(3,1) = FiniteField::fromParts(0, 1);
    A.at(3,2) = FiniteField::fromParts(0, 1);
    A.at(3,3) = FiniteField::fromParts(0,
        (uint128_t(0x44b888f25d038975ULL) << 64) | uint128_t(0x6a2f44fbcfadb125ULL));
    A.at(3,4) = FiniteField::fromParts(0, 1);
    A.at(3,5) = FiniteField::fromParts(0, 1);
    A.at(3,6) = FiniteField::fromParts(0, 1);
    A.at(3,7) = FiniteField::fromParts(0, 1);

    // 第五行
    A.at(4,0) = FiniteField::fromParts(0,
        (uint128_t(0x10aaa5e66d4bf298ULL) << 64) | uint128_t(0x0898dbe7d793f7e3ULL));
    A.at(4,1) = FiniteField::fromParts(0, 1);
    A.at(4,2) = FiniteField::fromParts(0, 1);
    A.at(4,3) = FiniteField::fromParts(0, 1);
    A.at(4,4) = FiniteField::fromParts(0,
        (uint128_t(0x56159fc662bdc559ULL) << 64) | uint128_t(0x79c12394208615baULL));
    A.at(4,5) = FiniteField::fromParts(0, 1);
    A.at(4,6) = FiniteField::fromParts(0, 1);
    A.at(4,7) = FiniteField::fromParts(0, 1);

    // 第六行
    A.at(5,0) = FiniteField::fromParts(0,
        (uint128_t(0x71589a612e203aa7ULL) << 64) | uint128_t(0xf60f4b94b8ebe913ULL));
    A.at(5,1) = FiniteField::fromParts(0, 1);
    A.at(5,2) = FiniteField::fromParts(0, 1);
    A.at(5,3) = FiniteField::fromParts(0, 1);
    A.at(5,4) = FiniteField::fromParts(0, 1);
    A.at(5,5) = FiniteField::fromParts(0,
        (uint128_t(0x23b1edef7c5da997ULL) << 64) | uint128_t(0x7b16937e5ee4fe7aULL));
    A.at(5,6) = FiniteField::fromParts(0, 1);
    A.at(5,7) = FiniteField::fromParts(0, 1);

    // 第七行
    A.at(6,0) = FiniteField::fromParts(0,
        (uint128_t(0x333e74c82500dd9dULL) << 64) | uint128_t(0x028b38301b424832ULL));
    A.at(6,1) = FiniteField::fromParts(0, 1);
    A.at(6,2) = FiniteField::fromParts(0, 1);
    A.at(6,3) = FiniteField::fromParts(0, 1);
    A.at(6,4) = FiniteField::fromParts(0, 1);
    A.at(6,5) = FiniteField::fromParts(0, 1);
    A.at(6,6) = FiniteField::fromParts(0,
        (uint128_t(0x5d760031af55a30eULL) << 64) | uint128_t(0x9f7d707b5e6281ecULL));
    A.at(6,7) = FiniteField::fromParts(0, 1);

    // 第八行
    A.at(7,0) = FiniteField::fromParts(0,
        (uint128_t(0x762e70586c76c981ULL) << 64) | uint128_t(0x2474e7dad0854a61ULL));
    A.at(7,1) = FiniteField::fromParts(0, 1);
    A.at(7,2) = FiniteField::fromParts(0, 1);
    A.at(7,3) = FiniteField::fromParts(0, 1);
    A.at(7,4) = FiniteField::fromParts(0, 1);
    A.at(7,5) = FiniteField::fromParts(0, 1);
    A.at(7,6) = FiniteField::fromParts(0, 1);
    A.at(7,7) = FiniteField::fromParts(0,
        (uint128_t(0x0ac4a1494b13bb5bULL) << 64) | uint128_t(0x67b248460fdc2274ULL));

    // 打印矩阵信息
    std::cout << "Testing 8x8 Matrix:\n";
    std::cout << A.toString();
    
    std::cout << "Determinant = ";
    A.determinant().print(std::cout);
    std::cout << std::endl;
    
    std::cout << "Is invertible: " << (A.isInvertible() ? "Yes" : "No") << "\n\n";
}

int main() {
    try {
        testMatrixInvertibility();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}

