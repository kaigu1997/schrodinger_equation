#ifndef MATRIX_H
#define MATRIX_H

// the header file to deal with rowwise 
// real(double) or complex (double) matrices
// stored in 1D array, calculated by
// Vector Mathematical Functions in MKL
// multiply with vector/matrix could be done via BLAS
// eigenvalue/eigenvector could be gotten from LAPACK
// Error code: 100 for different size

#include <complex>
#include <mkl.h>
#include <utility>
using namespace std;

// equivalent to MKL_Complex16
typedef complex<double> Complex;
// to access to matrix elements
typedef pair<int, int> Index;

// the mode constant for VMF functions in MKL
const MKL_INT64 mode = VML_HA;

// declaration of the two kinds of matrix
class RealMatrix;
class ComplexMatrix;

// transfrom a double array to a complex array using lambda
inline void real_to_complex(const double* da, Complex* ca, const int length)
{
    generate(ca, ca + length, [i = 0, da](void)mutable->Complex{ return da[i++]; });
}

// declaration of functions that have
// something to do with both kinds of matrix
ComplexMatrix operator*(const RealMatrix& lhs, const Complex& rhs);
ComplexMatrix operator*(const Complex& lhs, const RealMatrix& rhs);
ComplexMatrix operator/(const RealMatrix& lhs, const Complex& rhs);


// the functions of real matrix
class RealMatrix
{
private:
    int length;
    int nelements;
    double* content;
public:
    // default constructor with all zero
    RealMatrix(const int size);
    // copy constructor
    RealMatrix(const RealMatrix& matrix);
    // quasi copy constructor
    RealMatrix(const int size, const double* array);
    // one element is give number and the other are all zero
    RealMatrix(const int size, const Index& idx, const double& val);
    // destructor
    ~RealMatrix(void);
    // the size of the matrix
    int length_of_matrix(void) const;
    // direct access to internal data
    double* data(void);
    const double* data(void) const;
    // copy to an array
    void transform_to_1d(double* array) const;
    // overload operator(): return the element (=[][])
    double& operator()(const int& idx1, const int& idx2);
    double& operator()(const Index& idx);
    const double& operator()(const int& idx1, const int& idx2) const;
    const double& operator()(const Index& idx) const;
    // overload numerical calculation by VMF
    friend RealMatrix operator+(const RealMatrix& lhs, const RealMatrix& rhs);
    RealMatrix& operator+=(const RealMatrix& rhs);
    friend RealMatrix operator-(const RealMatrix& lhs, const RealMatrix& rhs);
    RealMatrix& operator-=(const RealMatrix& rhs);
    friend RealMatrix operator*(const RealMatrix& lhs, const double& rhs);
    friend RealMatrix operator*(const double& lhs, const RealMatrix& rhs);
    friend ComplexMatrix operator*(const RealMatrix& lhs, const Complex& rhs);
    friend ComplexMatrix operator*(const Complex& lhs, const RealMatrix& rhs);
    RealMatrix& operator*=(const double& rhs);
    friend RealMatrix operator/(const RealMatrix& lhs, const double& rhs);
    friend ComplexMatrix operator/(const RealMatrix& lhs, const Complex& rhs);
    RealMatrix& operator/=(const double& rhs);
    // assignment operator
    RealMatrix& operator=(const RealMatrix& rhs);
    RealMatrix& operator=(const double* array);
    // the two kinds of matrix could access to each other
    friend class ComplexMatrix;
};

// the functions of complex matrix, similar to above
class ComplexMatrix
{
private:
    int length;
    int nelements;
    Complex* content;
public:
    // default constructor with all zero
    ComplexMatrix(const int size);
    // copy constructor
    ComplexMatrix(const ComplexMatrix& matrix);
    // quasi copy constructor
    ComplexMatrix(const int size, const Complex* array);
    // copy constructor from real matrix
    ComplexMatrix(const RealMatrix& matrix);
    // quasi copy constructor from real matrix
    ComplexMatrix(const int size, const double* array);
    // one element is give number and the other are all zero
    ComplexMatrix(const int size, const Index& idx, const Complex& val);
    // destructor
    ~ComplexMatrix(void);
    // the size of the matrix
    int length_of_matrix(void) const;
    // direct access to internal data
    Complex* data(void);
    const Complex* data(void) const;
    // copy to an array
    void transform_to_1d(Complex* array) const;
    // overload operator(): return the element (=[][])
    Complex& operator()(const int& idx1, const int& idx2);
    Complex& operator()(const Index& idx);
    const Complex& operator()(const int& idx1, const int& idx2) const;
    const Complex& operator()(const Index& idx) const;
    // overload numerical calculation
    friend ComplexMatrix operator+(const ComplexMatrix& lhs, const ComplexMatrix& rhs);
    ComplexMatrix& operator+=(const ComplexMatrix& rhs);
    friend ComplexMatrix operator-(const ComplexMatrix& lhs, const ComplexMatrix& rhs);
    ComplexMatrix& operator-=(const ComplexMatrix& rhs);
    friend ComplexMatrix operator*(const RealMatrix& lhs, const Complex& rhs);
    friend ComplexMatrix operator*(const Complex& lhs, const RealMatrix& rhs);
    friend ComplexMatrix operator*(const ComplexMatrix& lhs, const Complex& rhs);
    friend ComplexMatrix operator*(const Complex& lhs, const ComplexMatrix& rhs);
    ComplexMatrix& operator*=(const Complex& rhs);
    friend ComplexMatrix operator/(const RealMatrix& lhs, const Complex& rhs);
    friend ComplexMatrix operator/(const ComplexMatrix& lhs, const Complex& rhs);
    ComplexMatrix& operator/=(const Complex& rhs);
    // assignment operator
    ComplexMatrix& operator=(const ComplexMatrix& rhs);
    ComplexMatrix& operator=(const Complex* array);
    // the two kinds of matrix could access to each other
    friend class RealMatrix;
};

#endif // !MATRIX_H
