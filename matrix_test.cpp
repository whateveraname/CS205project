#include <gtest/gtest.h>
#include <complex>
#include "Matrix.h"
#include "MyInt.h"

TEST(TypeTest, Int) {
    int data[2][2]{{1, 2}, {3, 4}};
    Matrix<int> m(2, 2, (void*)data);
    ASSERT_EQ(m.get(1, 1), 4);
}

TEST(TypeTest, Double){

}

TEST(TypeTest, Complex){

}



int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
