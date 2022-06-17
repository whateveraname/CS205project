/*naive version, no shared memory, all hard copy*/
#include <cstddef>
#include <complex>
#include <iostream>
#include <memory.h>
#include "Matrix.h"
#include "MatrixException.cpp"

//constructors
template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols) {
    rows_num = rows;
    cols_num = cols;
    data = new T[rows_num * cols_num];
}
template <class T>
Matrix<T>::Matrix(size_t rows, size_t cols, void* d) {
    rows_num = rows;
    cols_num = cols;
    data = new T[rows_num * cols_num];
    memcpy(data, d, sizeof(T) * rows_num * cols_num);
}

//copy constructor
template <class T>
Matrix<T>::Matrix(const Matrix& m) {
    if (m.rows_num != rows_num || m.cols_num != cols_num) {
        throw InvalidSizeException(rows_num, cols_num, m.rows_num, m.cols_num);
    }
    rows_num = m.rows_num;
    cols_num = m.cols_num;
    data = new T[rows_num * cols_num];
    memcpy(data, m.data, sizeof(T) * rows_num * cols_num);
}

//copy assignment
template <class T>
Matrix<T>& Matrix<T>::operator=(const Matrix& m) {
    if (m.rows_num != rows_num || m.cols_num != cols_num) {
        throw InvalidSizeException(rows_num, cols_num, m.rows_num, m.cols_num);
    }
    if (this == &m) return *this;
    delete[] data;
    rows_num = m.rows_num;
    cols_num = m.cols_num;
    data = new T[rows_num * cols_num];
    memcpy(data, m.data, sizeof(T) * rows_num * cols_num);
    return *this;
}

//destructor
template <class T>
Matrix<T>::~Matrix() {
    delete[] data;
}

//getter
template <class T>
T Matrix<T>::get(size_t row, size_t col) { // row and col start from 0
    if (row < 0) throw MatrixOutOfBoundException(0, row);
    if (row >= rows_num) throw MatrixOutOfBoundException(rows_num - 1, row);
    if (col < 0) throw MatrixOutOfBoundException(0, col);
    if (col >= cols_num) throw MatrixOutOfBoundException(cols_num - 1, col);
    return data[cols_num * row + col];
}

//setter
template <class T>
void Matrix<T>::set(size_t row, size_t col, T val) { // row and col start from 0
    if (row < 0) throw MatrixOutOfBoundException(0, row);
    if (row >= rows_num) throw MatrixOutOfBoundException(rows_num - 1, row);
    if (col < 0) throw MatrixOutOfBoundException(0, col);
    if (col >= cols_num) throw MatrixOutOfBoundException(cols_num - 1, col);
    data[cols_num * row + col] = val;
}

//finding maximum
template <class T>
T Matrix<T>::max(size_t start_row, size_t end_row, size_t start_col, size_t end_col) {
    if (start_row > end_row || start_col > end_col) throw InvalidParameterException();
    try {
        T max = get(start_row, start_col);
        for (size_t i = start_row; i <= end_row; i++) {
            for (size_t j = start_col; j <= end_col; j++) {
                get(i, j) > max ? max = get(i, j) : max;
            }
        }
    } catch (MatrixOutOfBoundException& e) {
        throw e;
    }
    return max;
}
template <class T>
T Matrix<T>::max() {
    try {
        return max(0, rows_num - 1, 0, cols_num - 1);
    } catch (MatrixException &e) {
        throw e;
    }
}
template <class T>
T Matrix<T>::max(size_t index, size_t dir) { //if dir = 0, row-specific; else col-specific
    try {
        if (dir) {
            return max(0, rows_num - 1, index, index);
        } else {
            return max(index, index, 0, cols_num - 1);
        }
    } catch (MatrixException& e) {
        throw e;
    }
}

//finding minimum
template <class T>
T Matrix<T>::min(size_t start_row, size_t end_row, size_t start_col, size_t end_col) {
    if (start_row > end_row || start_col > end_col) throw InvalidParameterException();
    try {
        T min = get(start_row, start_col);
        for (size_t i = start_row; i <= end_row; i++) {
            for (size_t j = start_col; j <= end_col; j++) {
                get(i, j) < min ? min = get(i, j) : min;
            }
        }
    } catch (MatrixOutOfBoundException& e) {
        throw e;
    }
    return min;
}
template <class T>
T Matrix<T>::min() {
    try {
        return min(0, rows_num - 1, 0, cols_num - 1);
    } catch (MatrixException& e) {
        throw e;
    }
}
template <class T>
T Matrix<T>::min(size_t index, size_t dir) { //if dir = 0, row-specific; else col-specific
    try {
        if (dir) {
            return min(0, rows_num - 1, index, index);
        } else {
            return min(index, index, 0, cols_num - 1);
        }
    } catch (MatrixException& e) {
        throw e;
    }
}

//summing
template <class T>
T Matrix<T>::sum(size_t start_row, size_t end_row, size_t start_col, size_t end_col) {
    if (start_row > end_row || start_col > end_col) throw InvalidParameterException();
    try {
        T sum = 0;
        for (size_t i = start_row; i <= end_row; i++) {
            for (size_t j = start_col; j <= end_col; j++) {
                sum = sum + get(i, j);
            }
        }
    } catch (MatrixOutOfBoundException& e) {
        throw e;
    }
    return sum;
}
template <class T>
T Matrix<T>::sum() {
    try {
        return sum(0, rows_num - 1, 0, cols_num - 1);
    } catch (MatrixException& e) {
        throw e;
    }
}
template <class T>
T Matrix<T>::sum(size_t index, size_t dir) { //if dir = 0, row-specific; else col-specific
    try {
        if (dir) {
            return sum(0, rows_num - 1, index, index);
        } else {
            return sum(index, index, 0, cols_num - 1);
        }
    } catch (MatrixException& e) {
        throw e;
    }
}

//averaging
template <class T>
T Matrix<T>::avg(size_t start_row, size_t end_row, size_t start_col, size_t end_col) {
    try {
        return sum(start_row, end_row, start_col, end_col) / ((end_row - start_row + 1) * (end_col - start_col + 1));
    } catch (MatrixException& e) {
        throw e;
    }
}
template <class T>
T Matrix<T>::avg() {
    try {
        return avg(0, rows_num - 1, 0, cols_num - 1);
    } catch (MatrixException& e) {
        throw e;
    }
}
template <class T>
T Matrix<T>::avg(size_t index, size_t dir) { //if dir = 0, row-specific; else col-specific
    try {
        if (dir) {
            return avg(0, rows_num - 1, index, index);
        } else {
            return avg(index, index, 0, cols_num - 1);
        }
    } catch (MatrixOutOfBoundException& e) {
        throw e;
    }
}

//eigenvalues and eigenvectors
template <class T>
T norm(T* a, size_t n) {
    T sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * a[i];
    }
    return (T)sqrt(sum);
}
template <class T>
void QR(Matrix<T>& A, Matrix<T>& Q, Matrix<T>& R) {
    size_t n = A.rows_num;
    T a[n], b[n];
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            a[i] = b[i] = A.get(i, j);
        }
        for (int k = 0; k < j; ++k) {
            R.set(k, j, 0);
            for (int l = 0; l < n; ++l) {
                R.set(k, j, R.get(k, j) + a[l] * Q.get(l, k));
            }
            for (int l = 0; l < n; ++l) {
                b[l] -= R.get(k, j) * Q.get(l, k);
            }
        }
        T norm = norm(b, n);
        R.set(j, j, norm);
        for (int i = 0; i < n; ++i) {
            Q.set(i, j, b[i] / norm);
        }
    }
}
template <class T>
void Matrix<T>::eig(T* eigen_values, Matrix<T>& eigen_vectors) {
    if (rows_num != cols_num) throw NotASquareMatrixException();
    try {
        size_t n = rows_num;
        Matrix<T> Q(n, n), R(n, n), temp = *this;
        size_t iter_num = n;
        for (int i = 0; i < iter_num; ++i) {
            QR(temp, Q, R);
            temp = R * Q;
        }
        for (int i = 0; i < n; ++i) {
            eigen_values[i] = temp.get(i, i);
        }
        eigen_vectors = Q;
    } catch (InvalidSizeException& e) {
        throw e;
    }
}

//traces
template <class T>
T Matrix<T>::trace() {
    if (rows_num != cols_num) throw NotASquareMatrixException();
    size_t size = rows_num;
    T res = 0;
    for (size_t i = 0; i < size; ++i) {
        res = res + get(i, i);
    }
    return res;
}

template <class T>
Matrix<T> Matrix<T>::company_matrix() {
    size_t size = rows_num;
    Matrix<T> result(size, size);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            Matrix<T> temp(size - 1, size - 1);
            for (size_t k = 0; k < size - 1; ++k) {
                for (size_t l = 0; l < size - 1; ++l) {
                    temp.set(i, j, get(k < i ? k : k + 1, l < j ? l : l + 1));
                }
            }
            result.set(i, j, temp.determinant() * ((i + j) % 2 ? -1 : 1));
        }
    }
    return transpose(result);
}

//determinant
template <class T>
T Matrix<T>::determinant() {
    if (rows_num != cols_num) throw NotASquareMatrixException();
    size_t size = rows_num;
    if (size == 1) return get(0, 0);
    T res = 0;
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            if (get(i, j)) {
                Matrix<T> temp(size - 1, size - 1);
                for (size_t k = 0; k < size - 1; ++k) {
                    for (size_t l = 0; l < size - 1; ++l) {
                        temp.set(i, j, get(k < i ? k : k + 1, l < j ? l : l + 1));
                    }
                }
                res = res + get(i, j) * temp.determinant() * ((i + j) % 2 ? -1 : 1);
            }
        }
    }
    return res;
}

template <class T>
Matrix<T> Matrix<T>::reshape(size_t row, size_t col){
    if (row * col != rows_num * cols_num) throw InvalidSizeException(rows_num, cols_num, row, col);
    return Matrix<T>(row, col, (void *)data);
}

template <class T>
Matrix<T> Matrix<T>::slice(size_t row_start, size_t row_end, size_t col_start, size_t col_end){    //start from 0, including start and end
    if (row_start < 0) throw MatrixOutOfBoundException(0, row_start);
    if (row_start >= rows_num) throw MatrixOutOfBoundException(rows_num - 1, row_start);
    if (col_start < 0) throw MatrixOutOfBoundException(0, col_start);
    if (col_start >= cols_num) throw MatrixOutOfBoundException(cols_num - 1, col_start);
    if (row_end < 0) throw MatrixOutOfBoundException(0, row_end);
    if (row_end >= rows_num) throw MatrixOutOfBoundException(rows_num - 1, row_end);
    if (col_end < 0) throw MatrixOutOfBoundException(0, col_end);
    if (col_end >= cols_num) throw MatrixOutOfBoundException(cols_num - 1, col_end);
    if (row_start > row_end || col_start > col_end) throw InvalidParameterException();
    size_t rows = row_end - row_start + 1;
    size_t cols = col_end - col_start + 1;
    T* d = new T[rows * cols];
    for(size_t i = row_start; i <= row_end; i++){
        memcpy(d + i * cols, data + i * cols_num + row_start, cols * sizeof(T));
    }
    Matrix<T> result(rows, cols, (void *)d);
    delete[] d;
    return result;  //sliced matrix
}

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
    if (m1.rows_num != m2.rows_num || m1.cols_num != m2.cols_num) throw OperandsSizeIncompatibleException();
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
    if (m1.rows_num != m2.rows_num || m1.cols_num != m2.cols_num) throw OperandsSizeIncompatibleException();
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
    if (m1.cols_num != m2.rows_num) throw OperandsSizeIncompatibleException();
    size_t row = m1.rows_num, col = m2.cols_num;
    Matrix<T> res(row, col);
    for (size_t i = 0; i < row; ++i) {
        for (size_t j = 0; j < col; ++j) {
            T temp = 0;
            for (size_t k = 0; k < m1.cols_num; ++k) {
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
    if (m1.rows_num != m2.rows_num || m1.cols_num != m2.cols_num) throw OperandsSizeIncompatibleException();
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
    if (v1.cols_num != v2.cols_num) throw OperandsSizeIncompatibleException();
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
    if (v1.cols_num != 3 || v2.cols_num != 3) throw InvalidParameterException();
    Matrix<T> res(1, 3);
    res.set(0, 0, v1.get(0, 1) * v2.get(0, 2) - v2.get(0, 1) * v1.get(0, 2));
    res.set(0, 1, v1.get(0, 2) * v2.get(0, 0) - v2.get(0, 2) * v1.get(0, 0));
    res.set(0, 0, v1.get(0, 0) * v2.get(0, 1) - v2.get(0, 0) * v1.get(0, 1));
    return res;
}

//inverse
template <class T>
Matrix<T> inverse(const Matrix<T> &m) {
    try {
        return m.company_matrix() / m.determinant();
    } catch (MatrixException& e) {
        throw e;
    }
}

//convolution
template<class T>
Matrix<T> conv(const Matrix<T>& m1, const Matrix<T>& m2){
    size_t rows = m1.rows_num + m2.rows_num - 1;
    size_t cols = m1.cols_num + m2.cols_num - 1;
    T* resData = new T[rows * cols];
    Matrix<T> res(rows, cols, (void *)resData);
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

//convert from opencv::Mat, we only implement the double type
Matrix<double> fromCV(cv::Mat &m) {
    return {(size_t)m.rows, (size_t)m.cols, (void *)m.data};
}

//convert to opencv::Mat, we only implement the double type
template<class T>
cv::Mat toCV(Matrix<T> &m) {
    return {m.rows_num, m.cols_num, CV_64F, (void *)m.data};
}