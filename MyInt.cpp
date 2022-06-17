#include "MyInt.h"

MyInt MyInt::operator+(MyInt &rhs) {
    return {data + rhs.data};
}

MyInt MyInt::operator-(MyInt &rhs) {
    return {data - rhs.data};
}

MyInt MyInt::operator/(MyInt &rhs) {
    return {data / rhs.data};
}

MyInt MyInt::operator*(MyInt &rhs) {
    return {data * rhs.data};
}

MyInt MyInt::operator+(int rhs) {
    return {data + rhs};
}

MyInt MyInt::operator-(int rhs) {
    return {data - rhs};
}

MyInt MyInt::operator*(int rhs) {
    return {data * rhs};
}

MyInt MyInt::operator/(int rhs) {
    return {data / rhs};
}

MyInt operator+(MyInt &lhs, MyInt &rhs) {
    return {lhs.data + rhs.data};
}

MyInt operator-(MyInt &lhs, MyInt &rhs) {
    return {lhs.data - rhs.data};
}

MyInt operator*(MyInt &lhs, MyInt &rhs) {
    return {lhs.data * rhs.data};
}

MyInt operator/(MyInt &lhs, MyInt &rhs) {
    return {lhs.data / rhs .data};
}

MyInt operator+(int lhs, MyInt &rhs) {
    return rhs + lhs;
}

MyInt operator-(int lhs, MyInt &rhs) {
    return rhs - lhs;
}

MyInt operator*(int lhs, MyInt &rhs) {
    return rhs * lhs;
}

MyInt operator/(int lhs, MyInt &rhs) {
    return rhs / lhs;
}
