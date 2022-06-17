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


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
