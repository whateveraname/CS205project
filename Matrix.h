#ifndef C___PROJECT_MATRIX_H
#define C___PROJECT_MATRIX_H
template <class T>
class Matrix {
public:
    size_t rows_num, cols_num;
    T *data;

    //constructors
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, T* d);

    //copy constructor
    Matrix(const Matrix& m);

    //copy assignment
    Matrix& operator=(const Matrix& m);

    //destructor
    ~Matrix();

    //getter
    T get(size_t row, size_t col);

    //setter
    void set(size_t row, size_t col, T val);

    //finding maximum
    T max();
    T max(size_t start_row, size_t end_row, size_t start_col, size_t end_col);
    T max(size_t index, size_t dir);

    //finding minimum
    T min();
    T min(size_t start_row, size_t end_row, size_t start_col, size_t end_col);
    T min(size_t index, size_t dir);

    //summing
    T sum();
    T sum(size_t start_row, size_t end_row, size_t start_col, size_t end_col);
    T sum(size_t index, size_t dir);

    //averaging
    T avg();
    T avg(size_t start_row, size_t end_row, size_t start_col, size_t end_col);
    T avg(size_t index, size_t dir);

    //eigenvalues eigenvectors
    void eig(T* eigen_values, Matrix<T>& eigen_vectors);

    //traces
    T trace();

    //determinant
    Matrix<T> company_matrix();
    T determinant();

    Matrix<T> reshape(size_t row, size_t col);

    Matrix<T> slice(size_t row_start, size_t row_end, size_t col_start, size_t col_end);

    friend std::ostream& operator<<(std::ostream& os, Matrix<T> m);
};

//matrix addition
template <class T>
Matrix<T> operator + (const Matrix<T> &m1, const Matrix<T> &m2);

//matrix subtraction
template <class T>
Matrix<T> operator - (const Matrix<T> &m1, const Matrix<T> &m2);

//scalar multiplication
template <class T>
Matrix<T> operator * (const Matrix<T> &m1, T c);
template <class T>
Matrix<T> operator * (T c, const Matrix<T> &m2);

//matrix multiplication
template <class T>
Matrix<T> operator * (const Matrix<T> &m1, const Matrix<T> &m2);

//element-wise multiplication
template <class T>
Matrix<T> element_wise_multiplication(const Matrix<T> &m1, const Matrix<T> &m2);

//scalar division
template <class T>
Matrix<T> operator / (const Matrix<T> &m1, T c);

//transposition
template <class T>
Matrix<T> transpose(const Matrix<T> &m);

//conjugation
template <class T>
Matrix< std::complex<T> > conjugate(const Matrix< std::complex<T> > &m);

//dot product
template <class T>
T dot(const Matrix<T> &v1, const Matrix<T> &v2);

//cross product
template <class T>
Matrix<T> cross(const Matrix<T> &v1, const Matrix<T> &v2);

//inverse
template <class T>
Matrix<T> inverse(const Matrix<T> &m);

template<class T>
Matrix<T> conv(const Matrix<T>& m1, const Matrix<T>& m2);
#endif //C___PROJECT_MATRIX_H
