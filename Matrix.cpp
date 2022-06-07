#include <iostream>
#include <memory.h>

/*naive version, no shared memory, all hard copy*/
template <class T>
class Matrix {
public:
    size_t rows_num, cols_num;
    T* data;

    //constructors
    Matrix(size_t rows, size_t cols, T* d) {
        rows_num = rows;
        cols_num = cols;
        data = new T[rows_num * cols_num];
        memcpy(data, d);
    }

    //copy constructor
    Matrix(const Matrix& m) {
        rows_num = m.rows_num;
        cols_num = m.cols_num;
        data = new T[rows_num * cols_num];
        memcpy(data, m.data);
    }

    //copy assignment
    Matrix& operator=(const Matrix& m) {
        if (this == &m) return *this;
        delete[] data;
        rows_num = m.rows_num;
        cols_num = m.cols_num;
        data = new T[rows_num * cols_num];
        memcpy(data, m.data);
        return *this;
    }

    //destructor
    ~Matrix() {
        delete[] data;
    }

    //getter
    T get(size_t row, size_t col) { // row and col start from 0
        return data[cols_num * row + col];
    }

    //setter
    void set(size_t row, size_t col, T val) { // row and col start from 0
        data[cols_num * row + col] = val;
    }

    //finding maximum
    T max(size_t start_row, size_t end_row, size_t start_col, size_t end_col) {
        T max = get(start_row, start_col);
        for (size_t i = start_row; i <= end_row; i++) {
            for (size_t j = start_col; j <= end_col; j++) {
                get(i, j) > max ? max = get(i, j) : max;
            }
        }
        return max;
    }
    T max() {
        return max(0, rows_num - 1, 0, cols_num - 1);
    }
    T max(size_t index, size_t dir) { //if dir = 0, row-specific; else col-specific
        if (dir) {
            return max(0, rows_num - 1, index, index);
        } else {
            return max(index, index, 0, cols_num - 1);
        }
    }

    //finding minimum
    T min(size_t start_row, size_t end_row, size_t start_col, size_t end_col) {
        T min = get(start_row, start_col);
        for (size_t i = start_row; i <= end_row; i++) {
            for (size_t j = start_col; j <= end_col; j++) {
                get(i, j) < min ? min = get(i, j) : min;
            }
        }
        return min;
    }
    T min() {
        return min(0, rows_num - 1, 0, cols_num - 1);
    }
    T min(size_t index, size_t dir) { //if dir = 0, row-specific; else col-specific
        if (dir) {
            return min(0, rows_num - 1, index, index);
        } else {
            return min(index, index, 0, cols_num - 1);
        }
    }

    //summing
    T sum(size_t start_row, size_t end_row, size_t start_col, size_t end_col) {
        T sum = 0;
        for (size_t i = start_row; i <= end_row; i++) {
            for (size_t j = start_col; j <= end_col; j++) {
                sum = sum + get(i, j);
            }
        }
        return sum;
    }
    T sum() {
        return sum(0, rows_num - 1, 0, cols_num - 1);
    }
    T sum(size_t index, size_t dir) { //if dir = 0, row-specific; else col-specific
        if (dir) {
            return sum(0, rows_num - 1, index, index);
        } else {
            return sum(index, index, 0, cols_num - 1);
        }
    }

    //averaging
    T avg(size_t start_row, size_t end_row, size_t start_col, size_t end_col) {
        return sum(start_row, end_row, start_col, end_col) / ((end_row - start_row + 1) * (end_col - start_col + 1));
    }
    T avg() {
        return avg(0, rows_num - 1, 0, cols_num - 1);
    }
    T avg(size_t index, size_t dir) { //if dir = 0, row-specific; else col-specific
        if (dir) {
            return avg(0, rows_num - 1, index, index);
        } else {
            return avg(index, index, 0, cols_num - 1);
        }
    }

    //eigenvalues

    //eigenvectors

    //traces

    //determinant
    T determinant() {

    }

    Matrix<T> reshape(size_t row, size_t col){
        return Matrix<T>(row, col, data);
    }

    Matrix<T> slice(size_t row_start, size_t row_end, size_t col_start, size_t col_end){    //start from 0, including start and end
        size_t rows = row_end - row_start + 1;
        size_t cols = col_end - col_start + 1;
        T* d = new T[rows * cols];
        for(size_t i = row_start; i <= row_end; i++){
            memcpy(d + i * cols, data + i * cols_num + row_start, cols);
        }
        return Matrix<T>(rows, cols, d);  //sliced matrix
    }

    friend std::ostream& operator<<(std::ostream& os, Matrix<T> m);
};

template<class T>
std::ostream &operator<<(std::ostream &os, Matrix<T> m) {
    os << "[";
    for (size_t i = 0; i < m.rows_num; i++){
        os << "[";
        for (size_t j = 0; j < m.cols_num - 1; j++){
            os << m.get(i, j) << ",\t";
        }
        os << m.get(i, m.cols_num - 1) << "]";
    }
    os << "]\n";
    return os;
}

//matrix addition
template <class T>
Matrix<T> operator + (const Matrix<T> &m1, const Matrix<T> &m2) {

}

//matrix substraction
template <class T>
Matrix<T> operator - (const Matrix<T> &m1, const Matrix<T> &m2) {

}

//scalar multiplication
template <class T>
Matrix<T> operator * (const Matrix<T> &m1, T c) {

}
template <class T>
Matrix<T> operator * (T c, const Matrix<T> &m2) {

}

//matrix multiplication
template <class T>
Matrix<T> operator * (const Matrix<T> &m1, const Matrix<T> &m2) {

}

//element-wise multiplication
template <class T>
Matrix<T> element_wise_multiplication(const Matrix<T> m1, const Matrix<T> m2) {

}

//scalar division
template <class T>
Matrix<T> operator / (const Matrix<T> &m1, T c) {

}

//transposition
template <class T>
Matrix<T> transpose(const Matrix<T> m) {

}

//conjugation
template <class T>
Matrix<T> conjugate(const Matrix<T> m) {

}

//dot product
template <class T>
Matrix<T> dot(const Matrix<T> v1, const Matrix<T> v2) {

}

//cross product
template <class T>
Matrix<T> cross(const Matrix<T> v1, const Matrix<T> v2) {

}

//inverse
template <class T>
Matrix<T> inverse(const Matrix<T> m) {

}