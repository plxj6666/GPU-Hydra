#include "matrix.h"

int main() {
    Matrix A(4, 4);
    A.at(0, 0) = FiniteField::fromParts(0, 1);
    A.at(0, 1) = FiniteField::fromParts(0, 2);
    A.at(0, 2) = FiniteField::fromParts(0, 3);
    A.at(0, 3) = FiniteField::fromParts(0, 4);
    A.at(1, 0) = FiniteField::fromParts(0, 5);
    A.at(1, 1) = FiniteField::fromParts(0, 6);
    A.at(1, 2) = FiniteField::fromParts(0, 7);
    A.at(1, 3) = FiniteField::fromParts(0, 8);
    A.at(2, 0) = FiniteField::fromParts(0, 9);
    A.at(2, 1) = FiniteField::fromParts(0, 10);
    A.at(2, 2) = FiniteField::fromParts(0, 11);
    A.at(2, 3) = FiniteField::fromParts(0, 12);
    A.at(3, 0) = FiniteField::fromParts(0, 13);
    A.at(3, 1) = FiniteField::fromParts(0, 14);
    A.at(3, 2) = FiniteField::fromParts(0, 15);
    A.at(3, 3) = FiniteField::fromParts(0, 16);

    Matrix B(4, 4);
    B = A;
    printf("B:\n%s\n", B.toString().c_str());

    Matrix C(4, 4);
    C = A.multiplyMatrices(A, B);
    printf("device:\n");
    printf("C:\n%s\n", C.toString().c_str());
    Matrix D(4, 4);
    D = A * B;
    printf("host:\n");
    printf("D:\n%s\n", D.toString().c_str());
    return 0;
}