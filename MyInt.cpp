#include "MyInt.h"

MyInt MyInt::operator+(const MyInt& rhs) const {
    return {data + rhs.data};
}

MyInt MyInt::operator-(const MyInt& rhs) const {
    return {data - rhs.data};
}

MyInt MyInt::operator/(const MyInt& rhs) const {
    return {data / rhs.data};
}

MyInt MyInt::operator*(const MyInt& rhs) const{
    return {data * rhs.data};
}

MyInt MyInt::operator+(int rhs) const {
    return {data + rhs};
}

MyInt MyInt::operator-(int rhs) const {
    return {data - rhs};
}

MyInt MyInt::operator*(int rhs) const {
    return {data * rhs};
}

MyInt MyInt::operator/(int rhs) const {
    return {data / rhs};
}

MyInt operator+(int lhs, const MyInt& rhs) {
    return rhs + lhs;
}

MyInt operator-(int lhs, const MyInt& rhs) {
    return rhs - lhs;
}

MyInt operator*(int lhs, const MyInt& rhs) {
    return rhs * lhs;
}

MyInt operator/(int lhs, const MyInt& rhs) {
    return rhs / lhs;
}
