cmake_minimum_required(VERSION 3.16.3)
project(C___project)

set(CMAKE_CXX_STANDARD 20)

add_executable(C___project Matrix.cpp MatrixException.cpp Matrix.h)

#find_package(GTest REQUIRED)
#include_directories(${GTEST_INCLUDE_DIRS})

enable_testing()
add_executable(
        matrix_test
        matrix_test.cpp
)
target_link_libraries(
        matrix_test
        gtest
        pthread
)

include(GoogleTest)
gtest_discover_tests(matrix_test)