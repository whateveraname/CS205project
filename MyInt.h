#ifndef C___PROJECT_MYINT_H
#define C___PROJECT_MYINT_H
#include <cstddef>

class MyInt{
public:
    MyInt():data(0){}

    size_t data;
    MyInt(size_t d):data(d){}
    MyInt operator+(const MyInt& rhs) const;
    MyInt operator-(const MyInt& rhs) const;
    MyInt operator*(const MyInt& rhs) const;
    MyInt operator/(const MyInt& rhs) const;

    MyInt operator+(int rhs) const;
    MyInt operator-(int rhs) const;
    MyInt operator*(int rhs) const;
    MyInt operator/(int rhs) const;

    friend MyInt operator+(int lhs, const MyInt& rhs);
    friend MyInt operator-(int lhs, const MyInt& rhs);
    friend MyInt operator*(int lhs, const MyInt& rhs);
    friend MyInt operator/(int lhs, const MyInt& rhs);
    
    bool operator==(const MyInt& rhs) const{
        return data==rhs.data;
    }
    
    bool operator!=(const MyInt& rhs) const{
        return data!=rhs.data;
    }
};



#endif //C___PROJECT_MYINT_H
