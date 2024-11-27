#include "matrix.h"
#include "polynomial.h"
#include <iostream>

int main() {
    // 创建一个4x4矩阵
    Matrix A(4, 4);
    
    // 初始化矩阵元素
    // 这里我们使用简单的数值来初始化矩阵
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            A.at(i, j) = FiniteField::fromParts(0, i + j + 1);
        }
    }
    
    // 打印矩阵
    std::cout << "Matrix A:" << std::endl;
    std::cout << A.toString() << std::endl;
    
    // 计算最小多项式
    Polynomial minPoly = A.getMinimalPolynomial();
    
    // 打印最小多项式
    std::cout << "Minimal Polynomial:" << std::endl;
    for (int i = 0; i <= minPoly.degree(); i++) {
        std::cout << "Coefficient of x^" << i << ": " << minPoly[i].toString() << std::endl;
    }
    
    // 验证最小多项式
    bool isValid = A.verifyMinimalPolynomial(minPoly);
    std::cout << "Is valid minimal polynomial: " << (isValid ? "Yes" : "No") << std::endl;
    
    return 0;
}
