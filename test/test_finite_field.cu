#include <iostream>
#include "finite_field.h"

void testAddition() {
    std::cout << "\n=== Testing Addition ===\n";
    // 测试用例1：原有的测试
    uint128_t a1_high = 0;
    uint128_t a1_low = (uint128_t(0x1517414516ULL) << 64) | uint128_t(0x00041c21cb8e1170ULL);
    
    uint128_t b1_high = 0;
    uint128_t b1_low = (uint128_t(0xd2e88cb2dcULL) << 64) | uint128_t(0x00291951f38cae60ULL);

    FiniteField f1 = FiniteField::fromParts(a1_high, a1_low);
    FiniteField f2 = FiniteField::fromParts(b1_high, b1_low);
    
    std::cout << "Test case 1:\n";
    std::cout << "f1 = ";
    f1.print(std::cout);
    std::cout << "\nf2 = ";
    f2.print(std::cout);
    std::cout << "\nf1 + f2 = ";
    (f1 + f2).print(std::cout);
    std::cout << "\n\n";

    // 测试用例3：接近模数的值
    uint128_t a3_low = (uint128_t(0x7fffffffffffffffULL) << 64);
    uint128_t b3_low = (uint128_t(0x7fffffffffffffffULL) << 64);
    
    FiniteField f5 = FiniteField::fromParts(0, a3_low);
    FiniteField f6 = FiniteField::fromParts(0, b3_low);
    
    std::cout << "Test case 3:\n";
    std::cout << "f5 = ";
    f5.print(std::cout);
    std::cout << "\nf6 = ";
    f6.print(std::cout);
    std::cout << "\nf5 + f6 = ";
    (f5 + f6).print(std::cout);
    std::cout << "\n\n";
}

void testMultiplication() {
    std::cout << "\n=== Testing Multiplication ===\n";
    
    // 测试用例1：原有的测试
    uint128_t a1_high = 0;
    uint128_t a1_low = (uint128_t(0x1517414516ULL) << 64) | uint128_t(0x00041c21cb8e1170ULL);
    
    uint128_t b1_high = 0;
    uint128_t b1_low = (uint128_t(0xd2e88cb2dcULL) << 64) | uint128_t(0x00291951f38cae60ULL);
    
    FiniteField f1 = FiniteField::fromParts(a1_high, a1_low);
    FiniteField f2 = FiniteField::fromParts(b1_high, b1_low);
    
    std::cout << "Test case 1:\n";
    std::cout << "f1 = ";
    f1.print(std::cout);
    std::cout << "\nf2 = ";
    f2.print(std::cout);
    std::cout << "\nExpected = 0x392c188e686152f42b394fa9929ab2d5\n";
    std::cout << "f1 * f2 = ";
    (f1 * f2).print(std::cout);
    std::cout << "\n\n";
    
    // 测试用例2：较小的数
    uint128_t a2_low = uint128_t(0x123456789abcdef0ULL);
    uint128_t b2_low = uint128_t(0xfedcba9876543210ULL);
    
    FiniteField f3 = FiniteField::fromParts(0, a2_low);
    FiniteField f4 = FiniteField::fromParts(0, b2_low);
    
    std::cout << "Test case 2:\n";
    std::cout << "f3 = ";
    f3.print(std::cout);
    std::cout << "\nf4 = ";
    f4.print(std::cout);
    std::cout << "\nf3 * f4 = ";
    (f3 * f4).print(std::cout);
    std::cout << "\n\n";
    
    // 测试用例3：接近模数的值
    uint128_t a3_low = (uint128_t(0x7fffffffffffffffULL) << 64);
    uint128_t b3_low = (uint128_t(0x7fffffffffffffffULL) << 64);
    
    FiniteField f5 = FiniteField::fromParts(0, a3_low);
    FiniteField f6 = FiniteField::fromParts(0, b3_low);
    
    std::cout << "Test case 3:\n";
    std::cout << "f5 = ";
    f5.print(std::cout);
    std::cout << "\nf6 = ";
    f6.print(std::cout);
    std::cout << "\nf5 * f6 = ";
    (f5 * f6).print(std::cout);
    std::cout << "\n\n";
    
    // 测试用例4：一个大数和一个小数
    uint128_t a4_low = (uint128_t(0x7fffffffffffffffULL) << 64);
    uint128_t b4_low = uint128_t(0x2ULL);
    
    FiniteField f7 = FiniteField::fromParts(0, a4_low);
    FiniteField f8 = FiniteField::fromParts(0, b4_low);
    
    std::cout << "Test case 4:\n";
    std::cout << "f7 = ";
    f7.print(std::cout);
    std::cout << "\nf8 = ";
    f8.print(std::cout);
    std::cout << "\nf7 * f8 = ";
    (f7 * f8).print(std::cout);
    std::cout << "\n";
}

void testDiagonalMultiplication() {
    std::cout << "\n=== Testing Diagonal Multiplication ===\n";
    
    // Matrix 2 的对角线元素
    uint128_t d1 = (uint128_t(0x56359e80e4c8291dULL) << 64) | uint128_t(0x42308c2d232bff6eULL);
    uint128_t d2 = (uint128_t(0x2a1923422426a5b0ULL) << 64) | uint128_t(0x7575e5fe4b4f37faULL);
    uint128_t d3 = (uint128_t(0x56497e02613398b3ULL) << 64) | uint128_t(0xd1e42a0223d02eb2ULL);
    uint128_t d4 = (uint128_t(0x6ffd0d47fc83baabULL) << 64) | uint128_t(0xd22f39256672bd73ULL);
    
    FiniteField f1 = FiniteField::fromParts(0, d1);
    FiniteField f2 = FiniteField::fromParts(0, d2);
    FiniteField f3 = FiniteField::fromParts(0, d3);
    FiniteField f4 = FiniteField::fromParts(0, d4);
    
    std::cout << "d1 = ";
    f1.print(std::cout);
    std::cout << "\n";
    
    FiniteField result = f1;
    std::cout << "After d1: ";
    result.print(std::cout);
    std::cout << "\n";
    
    result = result * f2;
    std::cout << "After d1*d2: ";
    result.print(std::cout);
    std::cout << "\n";
    
    result = result * f3;
    std::cout << "After d1*d2*d3: ";
    result.print(std::cout);
    std::cout << "\n";
    
    result = result * f4;
    std::cout << "Final d1*d2*d3*d4: ";
    // 在第3步计算时出现溢出，来源于商的不准确：
    //     === Testing Diagonal Multiplication ===
    // d1 = High: 0x00000000000000000000000000000000, Low: 0x56359e80e4c8291d42308c2d232bff6e
    // After d1: High: 0x00000000000000000000000000000000, Low: 0x56359e80e4c8291d42308c2d232bff6e
    // After d1*d2: High: 0x00000000000000000000000000000000, Low: 0x21ae4b90bb1a3f57cedbe7ba7e825869
    // After d1*d2*d3: High: 0x00000000000000000000000000000000, Low: 0x76d19fd7cacc48cfb8260d0817e65418
    // Final d1*d2*d3*d4: High: 0x00000000000000000000000000000000, Low: 0x356ae4e5e17cf087b492e9cfe926e4e9
    // Right d1*d2*d3 answer 0x76d19fd7cacc48cfb8260d0817e653be
    // Right d1*d2*d3*d4 answer 0x567439971b2d501fcff8d2a9e4d0585e
    result.print(std::cout);
    std::cout << "\n";
}

int main() {
    try {
        // testAddition();
        // testMultiplication();
        testDiagonalMultiplication();
        std::cout << "\nAll tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
