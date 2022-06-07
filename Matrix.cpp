/*naive version, no shared memory, all hard copy*/
#include <cstddef>
#include <complex>
#include <iostream>
#include <memory.h>
#include <algorithm>

template <class T>
class Matrix {
public:
    enum {FULL, SAME, VALID};
    size_t rows_num, cols_num;
    T* data;

    //constructors
    Matrix(size_t rows, size_t cols) {
        rows_num = rows;
        cols_num = cols;
        data = new T[rows_num * cols_num];
    }
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
    size_t row_n = m1.rows_num, col_n = m1.cols_num;
    Matrix<T> res(row_n, col_n);
    for (size_t i = 0; i < row_n; ++i) {
        for (size_t j = 0; j < col_n; ++j) {
            res.set(i, j, m1.get(i, j) + m2.get(i ,j));
        }
    }
    return res;
}

//matrix subtraction
template <class T>
Matrix<T> operator - (const Matrix<T> &m1, const Matrix<T> &m2) {
    size_t row_n = m1.rows_num, col_n = m1.cols_num;
    Matrix<T> res(row_n, col_n);
    for (size_t i = 0; i < row_n; ++i) {
        for (size_t j = 0; j < col_n; ++j) {
            res.set(i, j, m1.get(i, j) - m2.get(i ,j));
        }
    }
    return res;
}

//scalar multiplication
template <class T>
Matrix<T> operator * (const Matrix<T> &m1, T c) {
    size_t row_n = m1.rows_num, col_n = m1.cols_num;
    Matrix<T> res(row_n, col_n);
    for (size_t i = 0; i < row_n; ++i) {
        for (size_t j = 0; j < col_n; ++j) {
            res.set(i, j, m1.get(i, j) * c);
        }
    }
    return res;
}
template <class T>
Matrix<T> operator * (T c, const Matrix<T> &m2) {
    return m2 * c;
}

//matrix multiplication
template <class T>
Matrix<T> operator * (const Matrix<T> &m1, const Matrix<T> &m2) {
    size_t size = m1.rows_num;
    Matrix<T> res(size, size);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            T temp = 0;
            for (size_t k = 0; k < size; ++k) {
                temp = temp + m1.get(i, k) * m2.get(k, j);
            }
            res.set(i, j, temp);
        }
    }
    return res;
}

//element-wise multiplication
template <class T>
Matrix<T> element_wise_multiplication(const Matrix<T> &m1, const Matrix<T> &m2) {
    size_t row_n = m1.rows_num, col_n = m1.cols_num;
    Matrix<T> res(row_n, col_n);
    for (size_t i = 0; i < row_n; ++i) {
        for (size_t j = 0; j < col_n; ++j) {
            res.set(i, j, m1.get(i, j) * m2.get(i ,j));
        }
    }
    return res;
}

//scalar division
template <class T>
Matrix<T> operator / (const Matrix<T> &m1, T c) {
    size_t row_n = m1.rows_num, col_n = m1.cols_num;
    Matrix<T> res(row_n, col_n);
    for (size_t i = 0; i < row_n; ++i) {
        for (size_t j = 0; j < col_n; ++j) {
            res.set(i, j, m1.get(i, j) / c);
        }
    }
    return res;
}

//transposition
template <class T>
Matrix<T> transpose(const Matrix<T> &m) {
    size_t row_n = m.cols_num, col_n = m.rows_num;
    Matrix<T> res(row_n, col_n);
    for (size_t i = 0; i < row_n; ++i) {
        for (size_t j = 0; j < col_n; ++j) {
            res.set(i, j, m.get(j, i));
        }
    }
    return res;
}

//conjugation
template <class T>
Matrix< std::complex<T> > conjugate(const Matrix< std::complex<T> > &m) {
    Matrix< std::complex<T> > res(m.rows_num, m.cols_num);
    for (size_t i = 0; i < m.rows_num; ++i) {
        for (size_t j = 0; j < m.cols_num; ++j) {
            res.set(conj(m.get(i, j)));
        }
    }
    return res;
}

//dot product
template <class T>
T dot(const Matrix<T> &v1, const Matrix<T> &v2) {
    size_t dimension = v1.cols_num;
    T res = 0;
    for (size_t i = 0; i < dimension; ++i) {
        res  = res + v1.get(0, i) * v2.get(0, i);
    }
    return res;
}

//cross product
template <class T>
Matrix<T> cross(const Matrix<T> &v1, const Matrix<T> &v2) {
    Matrix<T> res(1, 3);
    res.set(0, 0, v1.get(0, 1) * v2.get(0, 2) - v2.get(0, 1) * v1.get(0, 2));
    res.set(0, 1, v1.get(0, 2) * v2.get(0, 0) - v2.get(0, 2) * v1.get(0, 0));
    res.set(0, 0, v1.get(0, 0) * v2.get(0, 1) - v2.get(0, 0) * v1.get(0, 1));
    return res;
}

//inverse
template <class T>
Matrix<T> inverse(const Matrix<T> &m) {
    size_t size = m.rows_num;
    Matrix<T> res(size, size, m), res_inverse(size, size), L(size, size), U(size, size), L_inverse(size, size), U_inverse(size, size);
    T s;
    for (size_t i = 0; i < size; ++i) { // L对角置1
        L.set(i, i, 1);
    }
    for (size_t i = 0; i < size; ++i) {
        U.set(0, i, m.get(0, i));
    }
    for (size_t i = 1; i < size; ++i) {
        L.set(i, 0, m.get(i, 0) / U.get(0, 0));
    }
    for (size_t i = 1; i < size; ++i) {
        for (size_t j = i; j < size; ++j) {
            T temp = 0;
            for (size_t k = 0; k < i; ++k) {
                temp = temp + L.get(i, k) * U.get(k, j);
            }
            U.set(i, j, m.get(i, j) - temp);
        }
        for (size_t j = i; j < size; ++j) {
            T temp = 0;
            for (size_t k = 0; k < i; ++k) {
                temp = temp + L.get(j, k) * U.get(k, i);
            }
            L.set(j, i, (m.get(j, i) - temp) / U.get(i, i));
        }
    }
    for (size_t j = 0; j < size; ++j) {
        for (size_t i = j; i < size; ++i) {
            if (i == j) {
                L_inverse.set(i, j, 1 / L.get(i, j));
            }
            if (i > j) {
                T temp = 0;
                for (size_t k = j; k < i; ++k) {
                    temp = temp + L.get(i, k) * L_inverse.get(k, j);
                }
                L_inverse.set(i, j, L_inverse.get(j, j) * (-temp));
            }
        }
    }
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = i; j >= 0; --j) {
            if (i == j) {
                U_inverse.set(j, i, 1 / U.get(j, i));
            }
            if (j < i) {
                T temp = 0;
                for (size_t k = j + 1; k <= i; ++k) {
                    temp = temp + U.get(j, k) * U_inverse.get(k, i);
                }
                U_inverse.set(j, i, -1 / U.get(j, j) * s);
            }
        }
    }
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            T temp = 0;
            for (size_t k = 0; k < size; ++k) {
                temp = temp + U_inverse.get(i, k) * L_inverse.get(k, j);
            }
            res_inverse.set(i, j, temp);
        }
    }
}

template<class T>
Matrix<T> conv(const Matrix<T>& m1, const Matrix<T>& m2, int mode){
    size_t rows = m1.rows_num + m2.rows_num - 1;
    size_t cols = m1.cols_num + m2.cols_num - 1;
    T* resData = new T[rows * cols];
    Matrix<T> res(rows, cols, resData);
    for (size_t i = 0; i < rows; i++){
        for (size_t j = 0; j < cols; j++){
            T temp = 0;
            for (size_t p = 0; p <= i; p++){
                for (size_t q = 0; q <= j; q++){
                    if (p < m2.rows_num && q < m2.cols_num && (i - p) < m1.rows_num && (j - q) < m1.cols_num){
                        temp += m1.get(i - p, j - q) * m2.get(p, q);
                    }
                }
            }
            res.set(i, j, temp);
        }
    }
    return res;
}