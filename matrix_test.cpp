#include <gtest/gtest.h>
#include <complex>
#include "Matrix.h"
#include "MyInt.h"
using namespace std;

TEST(TypeTest, Int) {//
    int data[2][2]{{1, 2}, {3, 4}};
    Matrix<int> m(2, 2, (void*)data);
    ASSERT_EQ(m.get(0, 0), 1);
    ASSERT_EQ(m.get(0, 1), 2);
    ASSERT_EQ(m.get(1, 0), 3);
    ASSERT_EQ(m.get(1, 1), 4);
}

TEST(TypeTest, Double) {
    double data[2][2]{{1.0,2.0},{3.0,4.0}};
    Matrix<double> m(2,2,(void*)data);
    ASSERT_EQ(m.get(0,0), 1.0);
    ASSERT_EQ(m.get(0,1), 2.0);
    ASSERT_EQ(m.get(1,0), 3.0);
    ASSERT_EQ(m.get(1,1), 4.0);
}

TEST(TypeTest, Complex) {
    complex<int> data[1][2]{{complex<int>(1,1), complex<int>(2,2)}};
    Matrix<complex<int>> m(1,2,(void*)data);
    ASSERT_EQ(m.get(0,0), complex<int>(1,1));
    ASSERT_EQ(m.get(0,1), complex<int>(2,2));
}

TEST(TypeTest, SelfDefined){
    MyInt data1[1][2]{{MyInt(1), MyInt(2)}};
    Matrix<MyInt> m1(1,2,(void*)data1);
    MyInt data2[2][1]{{MyInt(1)}, {MyInt(2)}};
    Matrix<MyInt> m2(2,1,(void*)data2);
    Matrix<MyInt> res(1,1,(void*)(new MyInt(5)));
    ASSERT_TRUE(res.equals(m1 * m2));
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

TEST(SliceTest, SliceTest){
    int data[3][4]{{1,2,3,4}, {5,6,7,8}, {9,10,11,12}};
    Matrix<int> m(3,4,(void*)data);
    int sd1[2][2]{{1,2},{5,6}};
    Matrix<int> s1(2,2,(void*)sd1);
    int sd2[3][3]{{2,3,4},{6,7,8},{10,11,12}};
    Matrix<int> s2(3,3,(void*)sd2);
    cout << m.slice(0,1,0,1);
    cout << m.slice(0,2,1,3);
    ASSERT_TRUE(s1.equals(m.slice(0,1,0,1)));
    ASSERT_TRUE(s2.equals(m.slice(0,2,1,3)));
}

TEST(ReshapeTest, ReshapeTest){
    int data[3][4]{{1,2,3,4}, {5,6,7,8}, {9,10,11,12}};
    Matrix<int> m(3,4,(void*)data);
    int rd1[2][6]{{1,2,3,4,5,6},{7,8,9,10,11,12}};
    Matrix<int> r1(2,6,(void*)rd1);
    int rd2[4][3]{{1,2,3},{4,5,6},{7,8,9},{10,11,12}};
    Matrix<int> r2(4,3,(void*)rd2);
    ASSERT_TRUE(r1.equals(m.reshape(2,6)));
    ASSERT_TRUE(r2.equals(m.reshape(4,3)));
}

TEST(ConvolutionTest, ConvolutionTest){
    int md1[3][3]{{1,2,3},{4,5,6},{7,8,9}};  //square matrix convolution
    int md2[2][5]{{1,2,3,4,5},{6,7,8,9,10}}; //arbitrary size
    Matrix<int> m1(3,3,(void*)md1);
    Matrix<int> m2(2,5,(void*)md2);
    int resd1[5][5]{
            {1,4,10,12,9},
            {8,26,56,54,36},
            {30,84,165,144,90},
            {56,134,236,186,108},
            {49,112,190,144,81}
    };
    int resd2[4][7]{
            {1,4,10,16,22,22,15},
            {10,32,68,89,110,96,60},
            {31,80,149,188,227,180,105},
            {42,97,166,190,214,161,90}
    };
    Matrix<int> res1(5,5,(void*)resd1);
    Matrix<int> res2(4,7,(void*)resd2);
    ASSERT_TRUE(res1.equals(conv(m1,m1)));
    ASSERT_TRUE(res2.equals(conv(m2,m1)));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
