#include <exception>
#include <string>
class MatrixException:public std::exception{
public:
    virtual std::string what() = 0;
};

class InvalidSizeException:public MatrixException{
private:
    size_t originalRow, originalCol, newRow, newCol;

public:
    InvalidSizeException(size_t oR, size_t oC, size_t nR, size_t nC): MatrixException(), originalRow(oR), originalCol(oC), newRow(nR), newCol(nC){}
    std::string what() override{
        return "Original size: " + std::to_string(originalRow) + "*" + std::to_string(originalCol) + ", new size: " +
                std::to_string(newRow) + "*" + std::to_string(newCol) + ".";
    }
};

class MatrixOutOfBoundException:public MatrixException{
private:
    size_t bound, index;

public:
    MatrixOutOfBoundException(size_t b, size_t i):bound(b), index(i){}
    std::string what() override{
        return "Index(" + std::to_string(index) + ") out of bound(" + std::to_string(bound) + ").";
    }
};