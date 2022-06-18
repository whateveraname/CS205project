/**
 * @mainpage Matrix Library Documentation
 * @authors
 * 12011319 陈言麒
 * 12012213 陈言麟
 */
#ifndef C___PROJECT_MATRIX_H
#define C___PROJECT_MATRIX_H

#include <opencv2/core/mat.hpp>
#include <complex>
#include <cstddef>
#include <iostream>
#include <memory.h>
#include "MatrixException.cpp"

/**
 * @brief Matrix class supporting all standard numeric types and custom numeric types.
 * @tparam T
 */
template<class T>
class Matrix {
public:
    /**
     * Size of the matrix.
     */
    size_t rows_num, cols_num;
    /**
     * Data in the matrix.
     */
    T *data;

    /**
     * Create a matrix with 0 row, 0 column and no data.
     * @brief Default constructor
     */
    Matrix():rows_num(0), cols_num(0){data = new T;};

    /**
     * Create a matrix with given row number and column number, and initialize all data to default value.
     * @brief Constructor
     * @param rows
     * @param cols
     */
    Matrix(size_t rows, size_t cols);

    /**
     * Create a matrix with given row number and column number, and set all data according to the given pointer to data.
     * @brief Constructor
     * @param rows
     * @param cols
     * @param d
     */
    Matrix(size_t rows, size_t cols, void *d);

    /**
     * Create a matrix by another matrix.
     * @brief Copy constructor
     * @param m
     */
    Matrix(const Matrix &m);

    /**
     * Assign a matrix to another matrix.
     * @brief Copy assignment operator
     * @param m
     * @return
     */
    Matrix &operator=(const Matrix &m);

    /**
     * @brief Destructor
     */
    ~Matrix();

    /**
     * @brief Get the value by row number and column number, indexes starting from 0.
     * @param row
     * @param col
     * @return A value in the matrix.
     */
    T get(size_t row, size_t col) const;

    /**
     * @brief Set the value by row number and column number, indexes starting from 0.
     * @param row
     * @param col
     * @param val
     */
    void set(size_t row, size_t col, T val);

    /**
     * @brief Find the maximum value in the whole matrix.
     * @return The maximum value.
     */
    T max();

    /**
     * @brief Find the maximum value in a specific area.
     * @param start_row The starting row number of the area.
     * @param end_row The ending row number of the area.
     * @param start_col The starting column number of the area.
     * @param end_col The ending column number of the area.
     * @return The maximum value.
     */
    T max(size_t start_row, size_t end_row, size_t start_col, size_t end_col);

    /**
     * @brief Find the maximum value in a specific row or column.
     * @param index The index of the row or column.
     * @param dir Specifies row or column, 0 for row, column otherwise.
     * @return The maximum value.
     */
    T max(size_t index, size_t dir);

    /**
     * @see max()
     */
    T min();

    /**
     * @see max(size_t start_row, size_t end_row, size_t start_col, size_t end_col)
     */
    T min(size_t start_row, size_t end_row, size_t start_col, size_t end_col);

    /**
     * @see max(size_t index, size_t dir)
     */
    T min(size_t index, size_t dir);

    /**
     * @brief The sum of all values in the matrix.
     * @return The sum.
     */
    T sum();

    /**
     * @brief The sum of values in a specific area.
     * @param start_row
     * @param end_row
     * @param start_col
     * @param end_col
     * @return The sum.
     */
    T sum(size_t start_row, size_t end_row, size_t start_col, size_t end_col);

    /**
     * @brief The sum of values in a row or column.
     * @param index
     * @param dir
     * @return The sum.
     */
    T sum(size_t index, size_t dir);

    /**
     * @brief The average value of all values in the matrix
     */
    T avg();

    /**
     * @brief The average value in a specific area.
     * @param start_row
     * @param end_row
     * @param start_col
     * @param end_col
     */
    T avg(size_t start_row, size_t end_row, size_t start_col, size_t end_col);

    /**
     * @brief The average value in a row or column.
     * @param index
     * @param dir
     */
    T avg(size_t index, size_t dir);

    /**
     * @brief Find the eigen value and the eigen vector of the matrix.
     * @param eigen_values An array to store the eigen values.
     * @param eigen_vectors An n * n matrix to store the eigen vectors, each column stores an eigen vector.
     * @throws NotASquareMatrixException If the matrix is not a square matrix.
     */
    void eig(T *eigen_values, Matrix<T> &eigen_vectors);

    /**
     * @brief Find the trace of the matrix.
     * @return The trace.
     */
    T trace();

    /**
     * @brief Find the company matrix of the matrix.
     * @return The company matrix.
     */
    Matrix<T> company_matrix() const;

    /**
     * @brief Find the determinant of the matrix.
     * @return The determinant.
     */
    T determinant() const;

    /**
     * @brief Reshape the matrix.
     * @param row The row number of the reshaped matrix.
     * @param col The column number of the reshaped matrix.
     * @return The reshaped matrix.
     * @throws InvalidSizeException If the number of values in the reshaped matrix is not equal to that of the original matrix.
     */
    Matrix<T> reshape(size_t row, size_t col);

    /**
     * @brief Slice the matrix.
     * @param row_start The starting row number of the sliced matrix.
     * @param row_end The ending row number of the sliced matrix.
     * @param col_start The starting column number of the sliced matrix.
     * @param col_end The ending column number of the sliced matrix.
     * @return The sliced matrix.
     * @throws MatrixOutOfBoundException
     */
    Matrix<T> slice(size_t row_start, size_t row_end, size_t col_start, size_t col_end);

    /**
     * @brief Override the << operator for outputting the matrix.
     */
    template<class C>
    friend std::ostream &operator<<(std::ostream &os, Matrix<C> m);

    /**
     * @brief Determine whether the row number, column number and all values of two matrices are exactly the same.
     */
    bool equals(const Matrix<T> &m);
};

/**
 * @brief Override the + operator for matrix addition.
 */
template<class T>
Matrix<T> operator+(const Matrix<T> &m1, const Matrix<T> &m2);

/**
 * @brief Override the - operator for matrix substraction.
 */
template<class T>
Matrix<T> operator-(const Matrix<T> &m1, const Matrix<T> &m2);

/**
 * @brief Override the * operator for scalar multiplication with .
 */
template<class T>
Matrix<T> operator*(const Matrix<T> &m1, T c);

/**
 * @brief Override the * operator for scalar multiplication.
 */
template<class T>
Matrix<T> operator*(T c, const Matrix<T> &m2);

//matrix multiplication
template<class T>
Matrix<T> operator*(const Matrix<T> &m1, const Matrix<T> &m2);

//element-wise multiplication
template<class T>
Matrix<T> element_wise_multiplication(const Matrix<T> &m1, const Matrix<T> &m2);

//scalar division
template<class T>
Matrix<T> operator/(const Matrix<T> &m1, T c);

//transposition
template<class T>
Matrix<T> transpose(const Matrix<T> &m);

//conjugation
template<class T>
Matrix<std::complex<T> > conjugate(const Matrix<std::complex<T> > &m);

//dot product
template<class T>
T dot(const Matrix<T> &v1, const Matrix<T> &v2);

//cross product
template<class T>
Matrix<T> cross(const Matrix<T> &v1, const Matrix<T> &v2);

//inverse
template<class T>
Matrix<T> inverse(const Matrix<T> &m);

//convolution
template<class T>
Matrix<T> conv(const Matrix<T> &m1, const Matrix<T> &m2);

//convert from opencv::Mat, we only implement the double type
Matrix<double> fromCV(cv::Mat &m);

//convert to opencv::Mat, we only implement the double type
cv::Mat toCV(Matrix<double> &m);

/*naive version, no shared memory, all hard copy*/
//constructors
template<class T>
Matrix<T>::Matrix(size_t rows, size_t cols) {
    rows_num = rows;
    cols_num = cols;
    data = new T[rows_num * cols_num];
    memset(data, 0, rows_num * cols_num * sizeof(T));
}

template<class T>
Matrix<T>::Matrix(size_t rows, size_t cols, void *d) {
    rows_num = rows;
    cols_num = cols;
    data = new T[rows_num * cols_num];
    memcpy(data, d, sizeof(T) * rows_num * cols_num);
}

//copy constructor
template<class T>
Matrix<T>::Matrix(const Matrix &m) {
    rows_num = m.rows_num;
    cols_num = m.cols_num;
    data = new T[rows_num * cols_num];
    memcpy(data, m.data, sizeof(T) * rows_num * cols_num);
}

//copy assignment
template<class T>
Matrix<T> &Matrix<T>::operator=(const Matrix &m) {
    if (this == &m) return *this;
    delete[] data;
    rows_num = m.rows_num;
    cols_num = m.cols_num;
    data = new T[rows_num * cols_num];
    memcpy(data, m.data, sizeof(T) * rows_num * cols_num);
    return *this;
}

//destructor
template<class T>
Matrix<T>::~Matrix() {
    delete[] data;
}

//getter
template<class T>
T Matrix<T>::get(size_t row, size_t col) const { // row and col start from 0
    if (row < 0) throw MatrixOutOfBoundException(0, row);
    if (row >= rows_num) throw MatrixOutOfBoundException(rows_num - 1, row);
    if (col < 0) throw MatrixOutOfBoundException(0, col);
    if (col >= cols_num) throw MatrixOutOfBoundException(cols_num - 1, col);
    return data[cols_num * row + col];
}

//setter
template<class T>
void Matrix<T>::set(size_t row, size_t col, T val) { // row and col start from 0
    if (row < 0) throw MatrixOutOfBoundException(0, row);
    if (row >= rows_num) throw MatrixOutOfBoundException(rows_num - 1, row);
    if (col < 0) throw MatrixOutOfBoundException(0, col);
    if (col >= cols_num) throw MatrixOutOfBoundException(cols_num - 1, col);
    data[cols_num * row + col] = val;
}

//finding maximum
template<class T>
T Matrix<T>::max(size_t start_row, size_t end_row, size_t start_col, size_t end_col) {
    if (start_row > end_row || start_col > end_col) throw InvalidParameterException();
    T res = get(start_row, start_col);
    for (size_t i = start_row; i <= end_row; i++) {
        for (size_t j = start_col; j <= end_col; j++) {
            get(i, j) > res ? res = get(i, j) : res;
        }
    }
    return res;
}

template<class T>
T Matrix<T>::max() {
    return max(0, rows_num - 1, 0, cols_num - 1);
}

template<class T>
T Matrix<T>::max(size_t index, size_t dir) { //if dir = 0, row-specific; else col-specific
    if (dir) {
        if (index < 0) throw MatrixOutOfBoundException(0, index);
        else if (index >= cols_num) throw MatrixOutOfBoundException(cols_num - 1, index);
        return max(0, rows_num - 1, index, index);
    } else {
        if (index < 0) throw MatrixOutOfBoundException(0, index);
        else if (index >= rows_num) throw MatrixOutOfBoundException(rows_num - 1, index);
        return max(index, index, 0, cols_num - 1);
    }
}

//finding minimum
template<class T>
T Matrix<T>::min(size_t start_row, size_t end_row, size_t start_col, size_t end_col) {
    if (start_row > end_row || start_col > end_col) throw InvalidParameterException();
    T min = get(start_row, start_col);
    for (size_t i = start_row; i <= end_row; i++) {
        for (size_t j = start_col; j <= end_col; j++) {
            get(i, j) < min ? min = get(i, j) : min;
        }
    }
    return min;
}

template<class T>
T Matrix<T>::min() {
    return min(0, rows_num - 1, 0, cols_num - 1);
}

template<class T>
T Matrix<T>::min(size_t index, size_t dir) { //if dir = 0, row-specific; else col-specific
    if (dir) {
        return min(0, rows_num - 1, index, index);
    } else {
        return min(index, index, 0, cols_num - 1);
    }
}

//summing
template<class T>
T Matrix<T>::sum(size_t start_row, size_t end_row, size_t start_col, size_t end_col) {
    if (start_row > end_row || start_col > end_col) throw InvalidParameterException();
    T sum = 0;
    for (size_t i = start_row; i <= end_row; i++) {
        for (size_t j = start_col; j <= end_col; j++) {
            sum = sum + get(i, j);
        }
    }
    return sum;
}

template<class T>
T Matrix<T>::sum() {
    return sum(0, rows_num - 1, 0, cols_num - 1);
}

template<class T>
T Matrix<T>::sum(size_t index, size_t dir) { //if dir = 0, row-specific; else col-specific
    if (dir) {
        return sum(0, rows_num - 1, index, index);
    } else {
        return sum(index, index, 0, cols_num - 1);
    }
}

//averaging
template<class T>
T Matrix<T>::avg(size_t start_row, size_t end_row, size_t start_col, size_t end_col) {
    return sum(start_row, end_row, start_col, end_col) / ((end_row - start_row + 1) * (end_col - start_col + 1));
}

template<class T>
T Matrix<T>::avg() {
    return avg(0, rows_num - 1, 0, cols_num - 1);
}

template<class T>
T Matrix<T>::avg(size_t index, size_t dir) { //if dir = 0, row-specific; else col-specific
    if (dir) {
        return avg(0, rows_num - 1, index, index);
    } else {
        return avg(index, index, 0, cols_num - 1);
    }
}

//eigenvalues and eigenvectors
template<class T>
T norm(T *a, size_t n) {
    T sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += a[i] * a[i];
    }
    return (T) sqrt(sum);
}

template<class T>
void QR(Matrix<T> &A, Matrix<T> &Q, Matrix<T> &R) {
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
        T norm2 = norm(b, n);
        R.set(j, j, norm2);
        for (int i = 0; i < n; ++i) {
            Q.set(i, j, b[i] / norm2);
        }
    }
}

template<class T>
void Matrix<T>::eig(T *eigen_values, Matrix<T> &eigen_vectors) {
    if (rows_num != cols_num) throw NotASquareMatrixException();
    size_t n = rows_num;
    Matrix<T> Q(n, n), R(n, n), temp = *this;
    for (int i = 0; i < n; ++i) {
        eigen_vectors.set(i, i, 1);
    }
    size_t iter_num = 100;
    for (int i = 0; i < iter_num; ++i) {
        QR(temp, Q, R);
        eigen_vectors = eigen_vectors * Q;
        temp = R * Q;
    }
    for (int i = 0; i < n; ++i) {
        eigen_values[i] = temp.get(i, i);
    }
}

//traces
template<class T>
T Matrix<T>::trace() {
    if (rows_num != cols_num) throw NotASquareMatrixException();
    size_t size = rows_num;
    T res = 0;
    for (size_t i = 0; i < size; ++i) {
        res = res + get(i, i);
    }
    return res;
}

template<class T>
Matrix<T> Matrix<T>::company_matrix() const {
    size_t size = rows_num;
    Matrix<T> result(size, size);
    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            Matrix<T> temp(size - 1, size - 1);
            for (size_t k = 0; k < size - 1; ++k) {
                for (size_t l = 0; l < size - 1; ++l) {
                    temp.set(k, l, get(k < i ? k : k + 1, l < j ? l : l + 1));
                }
            }
            result.set(i, j, temp.determinant() * ((i + j) % 2 ? -1 : 1));
        }
    }
    return transpose(result);
}

//determinant
template<class T>
T Matrix<T>::determinant() const {
    if (rows_num != cols_num) throw NotASquareMatrixException();
    size_t size = rows_num;
    if (size == 1) return get(0, 0);
    T res = 0;
    for (size_t j = 0; j < size; ++j) {
        if (get(0, j)) {
            Matrix<T> temp(size - 1, size - 1);
            for (size_t k = 0; k < size - 1; ++k) {
                for (size_t l = 0; l < size - 1; ++l) {
                    temp.set(k, l, get(k < 0 ? k : k + 1, l < j ? l : l + 1));
                }
            }
            res = res + get(0, j) * temp.determinant() * ((0 + j) % 2 ? -1 : 1);
        }
    }
    return res;
}

template<class T>
Matrix<T> Matrix<T>::reshape(size_t row, size_t col) {
    if (row * col != rows_num * cols_num) throw InvalidSizeException(rows_num, cols_num, row, col);
    return Matrix<T>(row, col, (void *) data);
}

template<class T>
Matrix<T> Matrix<T>::slice(size_t row_start, size_t row_end, size_t col_start,
                           size_t col_end) {    //start from 0, including start and end
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
    T *d = new T[rows * cols];
    for (size_t i = row_start; i <= row_end; i++) {
        memcpy(d + i * cols, data + i * cols_num + col_start, cols * sizeof(T));
    }
    Matrix<T> result(rows, cols, (void *) d);
    delete[] d;
    return result;  //sliced matrix
}

template<class T>
std::ostream &operator<<(std::ostream &os, Matrix<T> m) {
    for (size_t i = 0; i < m.rows_num; i++) {
        os << "[";
        for (size_t j = 0; j < m.cols_num - 1; j++) {
            os << m.get(i, j) << ", ";
        }
        os << m.get(i, m.cols_num - 1) << "]\n";
    }
    return os;
}

template<class T>
bool Matrix<T>::equals(const Matrix<T> &m) {
    if (rows_num != m.rows_num || cols_num != m.cols_num) return false;
    for (int i = 0; i < rows_num; ++i) {
        for (int j = 0; j < cols_num; ++j) {
            if (get(i, j) != m.get(i, j)) return false;
        }
    }
    return true;
}

//matrix addition
template<class T>
Matrix<T> operator+(const Matrix<T> &m1, const Matrix<T> &m2) {
    if (m1.rows_num != m2.rows_num || m1.cols_num != m2.cols_num) throw OperandsSizeIncompatibleException();
    size_t row_n = m1.rows_num, col_n = m1.cols_num;
    Matrix<T> res(row_n, col_n);
    for (size_t i = 0; i < row_n; ++i) {
        for (size_t j = 0; j < col_n; ++j) {
            res.set(i, j, m1.get(i, j) + m2.get(i, j));
        }
    }
    return res;
}

//matrix subtraction
template<class T>
Matrix<T> operator-(const Matrix<T> &m1, const Matrix<T> &m2) {
    if (m1.rows_num != m2.rows_num || m1.cols_num != m2.cols_num) throw OperandsSizeIncompatibleException();
    size_t row_n = m1.rows_num, col_n = m1.cols_num;
    Matrix<T> res(row_n, col_n);
    for (size_t i = 0; i < row_n; ++i) {
        for (size_t j = 0; j < col_n; ++j) {
            res.set(i, j, m1.get(i, j) - m2.get(i, j));
        }
    }
    return res;
}

//scalar multiplication
template<class T>
Matrix<T> operator*(const Matrix<T> &m1, T c) {
    size_t row_n = m1.rows_num, col_n = m1.cols_num;
    Matrix<T> res(row_n, col_n);
    for (size_t i = 0; i < row_n; ++i) {
        for (size_t j = 0; j < col_n; ++j) {
            res.set(i, j, m1.get(i, j) * c);
        }
    }
    return res;
}

template<class T>
Matrix<T> operator*(T c, const Matrix<T> &m2) {
    return m2 * c;
}

//matrix multiplication
template<class T>
Matrix<T> operator*(const Matrix<T> &m1, const Matrix<T> &m2) {
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
template<class T>
Matrix<T> element_wise_multiplication(const Matrix<T> &m1, const Matrix<T> &m2) {
    if (m1.rows_num != m2.rows_num || m1.cols_num != m2.cols_num) throw OperandsSizeIncompatibleException();
    size_t row_n = m1.rows_num, col_n = m1.cols_num;
    Matrix<T> res(row_n, col_n);
    for (size_t i = 0; i < row_n; ++i) {
        for (size_t j = 0; j < col_n; ++j) {
            res.set(i, j, m1.get(i, j) * m2.get(i, j));
        }
    }
    return res;
}

//scalar division
template<class T>
Matrix<T> operator/(const Matrix<T> &m1, T c) {
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
template<class T>
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
template<class T>
Matrix<std::complex<T> > conjugate(const Matrix<std::complex<T> > &m) {
    Matrix<std::complex<T> > res(m.rows_num, m.cols_num);
    for (size_t i = 0; i < m.rows_num; ++i) {
        for (size_t j = 0; j < m.cols_num; ++j) {
            res.set(i, j, conj(m.get(i, j)));
        }
    }
    return res;
}

//dot product
template<class T>
T dot(const Matrix<T> &v1, const Matrix<T> &v2) {
    if (v1.cols_num != v2.cols_num || v1.rows_num != 1 || v2.rows_num != 1) throw OperandsSizeIncompatibleException();
    size_t dimension = v1.cols_num;
    T res = 0;
    for (size_t i = 0; i < dimension; ++i) {
        res = res + v1.get(0, i) * v2.get(0, i);
    }
    return res;
}

//cross product
template<class T>
Matrix<T> cross(const Matrix<T> &v1, const Matrix<T> &v2) {
    if (v1.cols_num != 3 || v2.cols_num != 3 || v1.rows_num != 1 || v2.rows_num != 1) throw InvalidParameterException();
    Matrix<T> res(1, 3);
    res.set(0, 0, v1.get(0, 1) * v2.get(0, 2) - v2.get(0, 1) * v1.get(0, 2));
    res.set(0, 1, v1.get(0, 2) * v2.get(0, 0) - v2.get(0, 2) * v1.get(0, 0));
    res.set(0, 2, v1.get(0, 0) * v2.get(0, 1) - v2.get(0, 0) * v1.get(0, 1));
    return res;
}

//inverse
template<class T>
Matrix<T> inverse(const Matrix<T> &m) {
    return m.company_matrix() / m.determinant();
}

//convolution
template<class T>
Matrix<T> conv(const Matrix<T> &m1, const Matrix<T> &m2) {
    size_t rows = m1.rows_num + m2.rows_num - 1;
    size_t cols = m1.cols_num + m2.cols_num - 1;
    T *resData = new T[rows * cols];
    Matrix<T> res(rows, cols, (void *) resData);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            T temp = 0;
            for (size_t p = 0; p <= i; p++) {
                for (size_t q = 0; q <= j; q++) {
                    if (p < m2.rows_num && q < m2.cols_num && (i - p) < m1.rows_num && (j - q) < m1.cols_num) {
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
    return {(size_t) m.rows, (size_t) m.cols, (void *) m.data};
}

//convert to opencv::Mat, we only implement the double type
template<class T>
cv::Mat toCV(Matrix<T> &m) {
    return {m.rows_num, m.cols_num, CV_64F, (void *) m.data};
}

#endif //C___PROJECT_MATRIX_H
