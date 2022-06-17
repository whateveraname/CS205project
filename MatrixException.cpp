#include <exception>
#include <string>
class MatrixException : public std::exception {};

class InvalidSizeException : public MatrixException {
public:
    size_t originalRow, originalCol, newRow, newCol;

    InvalidSizeException(size_t oR, size_t oC, size_t nR, size_t nC): MatrixException(), originalRow(oR), originalCol(oC), newRow(nR), newCol(nC){}

    const char* what() {
        char* content = nullptr;
        strcpy(content, ("Original size: " + std::to_string(originalRow) + "*" + std::to_string(originalCol) + ", new size: " +
                         std::to_string(newRow) + "*" + std::to_string(newCol) + ".").c_str());
        return content;
    }
};

class MatrixOutOfBoundException : public MatrixException {
public:
    size_t bound, index;

    MatrixOutOfBoundException(size_t b, size_t i):bound(b), index(i){}

    const char* what() {
        char* content = nullptr;
        strcpy(content, ("Index(" + std::to_string(index) + ") out of bound(" + std::to_string(bound) + ").").c_str());
        return content;
    }
};

class InvalidParameterException : public MatrixException {
public:
    const char* what() {
        return "InvalidParameterException";
    }
};

class NotASquareMatrixException : public MatrixException {
public:
    const char* what() {
        return "The matrix is not a square matrix";
    }
};

class OperandsSizeIncompatibleException : public MatrixException {
public:
    const char* what() {
        return "The operands of the operator are incompatible";
    }
};