#include <gtest/gtest.h>
#include <complex>
#include "Matrix.h"
#include "MyInt.h"
using namespace std;

TEST(TypeTest, Int) {
    int data[2][2]{{1, 2}, {3, 4}};
    Matrix<int> m(2, 2, (void*)data);
    ASSERT_EQ(m.get(0, 0), 1);
}

TEST(TypeTest, Double) {

}

TEST(TypeTest, Complex) {

}

TEST(ArithmeticTest, addition) {
    int data1[2][2]{{1, 2}, {3, 4}};
    int data2[2][2]{{5, 6}, {7, 8}};
    Matrix<int> a(2, 2, (void*)data1);
    Matrix<int> b(2, 2, (void*)data2);
    int res_d[2][2]{{6, 8}, {10, 12}};
    Matrix<int> res(2, 2, (void*)res_d);
    ASSERT_TRUE(res.equals(a + b));
}

TEST(ArithmeticTest, subtraction) {
    int data1[2][2]{{1, 2}, {3, 4}};
    int data2[2][2]{{5, 6}, {7, 8}};
    Matrix<int> a(2, 2, (void*)data1);
    Matrix<int> b(2, 2, (void*)data2);
    int res_d[2][2]{{4, 4}, {4, 4}};
    Matrix<int> res(2, 2, (void*)res_d);
    ASSERT_TRUE(res.equals(b - a));
}

TEST(ArithmeticTest, matrix_mul) {
    int data1[2][2]{{1, 2}, {3, 4}};
    int data2[2][2]{{5, 6}, {7, 8}};
    Matrix<int> a(2, 2, (void*)data1);
    Matrix<int> b(2, 2, (void*)data2);
    int res_d[2][2]{{19, 22}, {43, 50}};
    Matrix<int> res(2, 2, (void*)res_d);
    ASSERT_TRUE(res.equals(a * b));
}

TEST(ArithmeticTest, scalar_mul) {
    int data1[2][2]{{1, 2}, {3, 4}};
    Matrix<int> a(2, 2, (void*)data1);
    int res_d[2][2]{{2, 4}, {6, 8}};
    Matrix<int> res(2, 2, (void*)res_d);
    ASSERT_TRUE(res.equals(a * 2));
    ASSERT_TRUE(res.equals(2 * a));
}

TEST(ArithmeticTest, elewise_mul) {
    int data1[2][2]{{1, 2}, {3, 4}};
    int data2[2][2]{{5, 6}, {7, 8}};
    Matrix<int> a(2, 2, (void*)data1);
    Matrix<int> b(2, 2, (void*)data2);
    int res_d[2][2]{{5, 12}, {21, 32}};
    Matrix<int> res(2, 2, (void*)res_d);
    ASSERT_TRUE(res.equals(element_wise_multiplication(a, b)));
}

TEST(ArithmeticTest, scalar_div) {
    int data1[2][2]{{2, 4}, {6, 8}};
    Matrix<int> a(2, 2, (void*)data1);
    int res_d[2][2]{{1, 2}, {3, 4}};
    Matrix<int> res(2, 2, (void*)res_d);
    ASSERT_TRUE(res.equals(a / 2));
}

TEST(TransformationTest, transpose) {
    int data1[2][2]{{2, 4}, {6, 8}};
    Matrix<int> a(2, 2, (void*)data1);
    int res_d[2][2]{{2, 6}, {4, 8}};
    Matrix<int> res(2, 2, (void*)res_d);
    ASSERT_TRUE(res.equals(transpose(a)));
}

TEST(TransformationTest, conjugate) {
    complex<int> data1[2][2]{{complex<int>(1, 2), complex<int>(3, 4)}, {complex<int>(5, 6), complex<int>(7, 8)}};
    Matrix<complex<int>> a(2, 2, (void*)data1);
    complex<int> res_d[2][2]{{complex<int>(1, -2), complex<int>(3, -4)}, {complex<int>(5, -6), complex<int>(7, -8)}};
    Matrix<complex<int>> res(2, 2, (void*)res_d);
    ASSERT_TRUE(res.equals(conjugate(a)));
}

TEST(TransformationTest, inverse) {
    int data[2][2]{{1,  2},
                   {-1, -3}};
    Matrix<int> a(2, 2, (void*)data);
    int res_d[2][2]{{3,  2},
                    {-1, -1}};
    Matrix<int> res(2, 2, (void*)res_d);
    ASSERT_TRUE(res.equals(inverse(a)));
}

TEST(VectorCalcTest, dot) {
    int d1[3]{1, 2, 3};
    int d2[3]{4, 5, 6};
    Matrix<int> v1(1, 3, (void*)d1);
    Matrix<int> v2(1, 3, (void*)d2);
    ASSERT_EQ(dot(v1, v2), 32);
}

TEST(VectorCalcTest, cross) {
    int d1[3]{1, 2, 3};
    int d2[3]{4, 5, 6};
    Matrix<int> v1(1, 3, (void*)d1);
    Matrix<int> v2(1, 3, (void*)d2);
    int res_d[3]{-3, 6, -3};
    Matrix<int> res(1, 3, (void*)res_d);
    ASSERT_TRUE(res.equals(cross(v1, v2)));
}

TEST(ReductionTest, max) {
    int data[2][2]{{1, 2},
                   {3, 4}};
    Matrix<int> a(2, 2, (void*)data);
    ASSERT_EQ(a.max(), 4);
    ASSERT_EQ(a.max(0, 0, 0, 1), 2);
    ASSERT_EQ(a.max(1, 1), 4);
}

TEST(ReductionTest, min) {
    int data[2][2]{{1, 2},
                   {3, 4}};
    Matrix<int> a(2, 2, (void*)data);
    ASSERT_EQ(a.min(), 1);
    ASSERT_EQ(a.min(0, 0, 0, 1), 1);
    ASSERT_EQ(a.min(1, 1), 2);
}

TEST(ReductionTest, sum) {
    int data[2][2]{{1, 2},
                   {3, 4}};
    Matrix<int> a(2, 2, (void*)data);
    ASSERT_EQ(a.sum(), 10);
    ASSERT_EQ(a.sum(0, 0, 0, 1), 3);
    ASSERT_EQ(a.sum(1, 1), 6);
}

TEST(ReductionTest, avg) {
    int data[2][2]{{4, 8},
                   {12, 16}};
    Matrix<int> a(2, 2, (void*)data);
    ASSERT_EQ(a.avg(), 10);
    ASSERT_EQ(a.avg(0, 0, 0, 1), 6);
    ASSERT_EQ(a.avg(1, 1), 12);
}

TEST(TestSuite5, eig) {
    double data[3][3]{{1, 0,  2},
                      {0, -1, 0},
                      {0, 4,  2}};
    Matrix<double> a(3, 3, (void *) data);
    double eigenvalues[3];
    double res[3]{1, 2, -1};
    Matrix<double> eigvecs(3, 3);
    a.eig(eigenvalues, eigvecs);
    cout << eigvecs << endl;
    ASSERT_EQ(eigenvalues[0], res[0]);
    ASSERT_EQ(eigenvalues[1], res[1]);
    ASSERT_EQ(eigenvalues[2], res[2]);
}

TEST(TestSuite5, trace) {
    int data[2][2]{{1,  2},
                   {-1, -3}};
    Matrix<int> a(2, 2, (void*)data);
    ASSERT_EQ(a.trace(), -2);
}

TEST(TestSuite5, determinant) {
    int data[2][2]{{1,  2},
                   {-1, -3}};
    Matrix<int> a(2, 2, (void*)data);
    ASSERT_EQ(a.determinant(), -1);
}

TEST(TestSuite5, company_matrix) {
    int data[2][2]{{1,  2},
                   {-1, -3}};
    Matrix<int> a(2, 2, (void*)data);
    int res_d[2][2]{{-3,  -2},
                    {1, 1}};
    Matrix<int> res(2, 2, (void*)res_d);
    ASSERT_TRUE(res.equals(a.company_matrix()));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
