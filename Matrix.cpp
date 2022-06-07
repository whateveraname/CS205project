/*naive version, no shared memory, all hard copy*/
template <class T>
class Matrix {
    int rows_num, cols_num;
    T* data;

    //constructors
    Matrix(int rows, int cols, T* d) {
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
    T get(int row, int col) { // row and col start from 0
        return data[cols_num * row + col];
    }

    //setter
    void set(int row, int col, T val) { // row and col start from 0
        data[cols_num * row + col] = val;
    }

    //finding maximum
    T max(int start_row, int end_row, int start_col, int end_col) {
        T max = get(start_row, start_col);
        for (int i = start_row; i <= end_row; i++) {
            for (int j = start_col; j <= end_col; j++) {
                get(i, j) > max ? max = get(i, j) : max;
            }
        }
        return max;
    }
    T max() {
        return max(0, rows_num - 1, 0, cols_num - 1);
    }
    T max(int index, int dir) { //if dir = 0, row-specofic; else col-specific
        if (dir) {
            return max(0, rows_num - 1, index, index);
        } else {
            return max(index, index, 0, cols_num - 1);
        }
    }

    //finding minimum
    T min(int start_row, int end_row, int start_col, int end_col) {
        T min = get(start_row, start_col);
        for (int i = start_row; i <= end_row; i++) {
            for (int j = start_col; j <= end_col; j++) {
                get(i, j) < min ? min = get(i, j) : min;
            }
        }
        return min;
    }
    T min() {
        return min(0, rows_num - 1, 0, cols_num - 1);
    }
    T min(int index, int dir) { //if dir = 0, row-specofic; else col-specific
        if (dir) {
            return min(0, rows_num - 1, index, index);
        } else {
            return min(index, index, 0, cols_num - 1);
        }
    }

    //summing
    T sum(int start_row, int end_row, int start_col, int end_col) {
        T sum = 0;
        for (int i = start_row; i <= end_row; i++) {
            for (int j = start_col; j <= end_col; j++) {
                sum = sum + get(i, j);
            }
        }
        return sum;
    }
    T sum() {
        return sum(0, rows_num - 1, 0, cols_num - 1);
    }
    T sum(int index, int dir) { //if dir = 0, row-specofic; else col-specific
        if (dir) {
            return sum(0, rows_num - 1, index, index);
        } else {
            return sum(index, index, 0, cols_num - 1);
        }
    }

    //averaging
    T avg(int start_row, int end_row, int start_col, int end_col) {
        return sum(start_row, end_row, start_col, end_col) / ((end_row - start_row + 1) * (end_col - start_col + 1));
    }
    T avg() {
        return avg(0, rows_num - 1, 0, cols_num - 1);
    }
    T avg(int index, int dir) { //if dir = 0, row-specofic; else col-specific
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
};

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