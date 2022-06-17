#ifndef C___PROJECT_MYINT_H
#define C___PROJECT_MYINT_H
#include <cstddef>

class MyInt{
public:
    size_t data;
    MyInt(size_t d):data(d){}
    MyInt operator+(MyInt& rhs);
    MyInt operator-(MyInt& rhs);
    MyInt operator*(MyInt& rhs);
    MyInt operator/(MyInt& rhs);

    MyInt operator+(int rhs);
    MyInt operator-(int rhs);
    MyInt operator*(int rhs);
    MyInt operator/(int rhs);

    friend MyInt operator+(MyInt& lhs, MyInt& rhs);
    friend MyInt operator-(MyInt& lhs, MyInt& rhs);
    friend MyInt operator*(MyInt& lhs, MyInt& rhs);
    friend MyInt operator/(MyInt& lhs, MyInt& rhs);

    friend MyInt operator+(int lhs, MyInt& rhs);
    friend MyInt operator-(int lhs, MyInt& rhs);
    friend MyInt operator*(int lhs, MyInt& rhs);
    friend MyInt operator/(int lhs, MyInt& rhs);
};



#endif //C___PROJECT_MYINT_H
