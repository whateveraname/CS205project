cmake_minimum_required(VERSION 3.16.3)
project(C___project)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(C___project MatrixException.cpp Matrix.h MyInt.cpp MyInt.h)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${GTEST_INCLUDE_DIRS})

enable_testing()
add_executable(
        matrix_test
        matrix_test.cpp
        MyInt.cpp MyInt.h)
target_link_libraries(
        matrix_test
        gtest
        pthread
)

include(GoogleTest)
gtest_discover_tests(matrix_test)