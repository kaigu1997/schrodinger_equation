// implemetation of matrix.h

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <utility>
#include <mkl.h>
#include "matrix.h"
using namespace std;

// transfrom a double array to a complex array 
// by using lambda expression
void real_to_complex(const double* da, Complex* ca, const int length)
{
    generate(ca, ca + length, [i = 0, da](void)mutable->Complex{ return da[i++]; });
}

// functions that have something to do with both kinds of matrix
ComplexMatrix operator*(const RealMatrix& lhs, const Complex& rhs)
{
	ComplexMatrix result(lhs.length);
	Complex* content = new Complex[lhs.nelements];
	real_to_complex(lhs.content, content, lhs.nelements);
	Complex* num = new Complex[lhs.nelements];
	fill(num, num + lhs.nelements, rhs);
	vmzMul(lhs.nelements, reinterpret_cast<const MKL_Complex16*>(content), reinterpret_cast<const MKL_Complex16*>(num), reinterpret_cast<MKL_Complex16*>(result.content), mode);
	return result;
}
ComplexMatrix operator*(const Complex& lhs, const RealMatrix& rhs)
{
	ComplexMatrix result(rhs.length);
	Complex* content = new Complex[rhs.nelements];
	real_to_complex(rhs.content, content, rhs.nelements);
	Complex* num = new Complex[rhs.nelements];
	fill(num, num + rhs.nelements, lhs);
	vmzMul(rhs.nelements, reinterpret_cast<const MKL_Complex16*>(content), reinterpret_cast<const MKL_Complex16*>(num), reinterpret_cast<MKL_Complex16*>(result.content), mode);
	return result;
}
ComplexMatrix operator/(const RealMatrix& lhs, const Complex& rhs)
{
	return lhs * (1.0 / rhs);
}


// RealMatrix functions
// default constructor with all zero
RealMatrix::RealMatrix(const int size)
	: length(size), nelements(length * length), content(new double[nelements])
{
	memset(content, 0, nelements * sizeof(double));
}

// copy constructor
RealMatrix::RealMatrix(const RealMatrix& matrix)
	: length(matrix.length), nelements(matrix.nelements), content(new double[nelements])
{
	copy(matrix.content, matrix.content + nelements, content);
}

// quasi copy constructor
RealMatrix::RealMatrix(const int size, const double* array)
	: length(size), nelements(length * length), content(new double[nelements])
{
	copy(array, array + nelements, content);
}

// move constructor
RealMatrix::RealMatrix(RealMatrix&& matrix)
	: length(move(matrix.length)), nelements(move(matrix.nelements)), content(move(matrix.content))
{
}

// one element is give number and the other are all zero
RealMatrix::RealMatrix(const int size, const Index& idx, const double& val)
	:length(size), nelements(length * length), content(new double[nelements])
{
	memset(content, 0, nelements * sizeof(double));
	content[idx.first * length + idx.second] = val;
}

// destructor
RealMatrix::~RealMatrix(void)
{
	delete[] content;
}

// the size of the matrix
int RealMatrix::length_of_matrix(void) const
{
	return length;
}

// direct access to internal data
double* RealMatrix::data(void)
{
	return content;
}
const double* RealMatrix::data(void) const
{
	return content;
}

// copy to an array
void RealMatrix::transform_to_1d(double* array) const
{
	copy(content, content + nelements, array);
}

// overload operator(): return the element (=[][])
double& RealMatrix::operator()(const int idx1, const int idx2)
{
	return content[idx1 * length + idx2];
}
double& RealMatrix::operator()(const Index& idx)
{
	return content[idx.first * length + idx.second];
}
const double& RealMatrix::operator()(const int idx1, const int idx2) const
{
	return content[idx1 * length + idx2];
}
const double& RealMatrix::operator()(const Index& idx) const
{
	return content[idx.first * length + idx.second];
}

// overload numerical calculation by VMF
RealMatrix operator+(const RealMatrix& lhs, const RealMatrix& rhs)
{
	if (lhs.length != rhs.length)
	{
		cerr << "ADD DIFFERENT SIZE REAL MATRIX" << endl;
		exit(100);
	}
	RealMatrix result(lhs.length);
	vmdAdd(result.nelements, lhs.content, rhs.content, result.content, mode);
	return result;
}

RealMatrix& RealMatrix::operator+=(const RealMatrix& rhs)
{
	if (length != rhs.length)
	{
		cerr << "ADD-ASSIGN DIFFERENT SIZE REAL MATRIX" << endl;
		exit(101);
	}
	double* result = new double[nelements];
	vmdAdd(nelements, content, rhs.content, result, mode);
	swap(content, result);
	delete[] result;
	return *this;
}
RealMatrix operator-(const RealMatrix& lhs, const RealMatrix& rhs)
{
	if (lhs.length != rhs.length)
	{
		cerr << "SUBTRACT DIFFERENT SIZE REAL MATRIX" << endl;
		exit(102);
	}
	RealMatrix result(lhs.length);
	vmdSub(result.nelements, lhs.content, rhs.content, result.content, mode);
	return result;
}
RealMatrix& RealMatrix::operator-=(const RealMatrix& rhs)
{
	if (length != rhs.length)
	{
		cerr << "SUBSTRACT-ASSIGN DIFFERENT SIZE REAL MATRIX" << endl;
		exit(103);
	}
	double* result = new double[nelements];
	vmdSub(nelements, content, rhs.content, result, mode);
	swap(content, result);
	delete[] result;
	return *this;
}
RealMatrix operator*(const RealMatrix& lhs, const double& rhs)
{
	RealMatrix result(lhs.length);
	double* num = new double[lhs.nelements];
	fill(num, num + lhs.nelements, rhs);
	vmdMul(lhs.nelements, lhs.content, num, result.content, mode);
	return result;
}
RealMatrix operator*(const double& lhs, const RealMatrix& rhs)
{
	RealMatrix result(rhs.length);
	double* num = new double[rhs.nelements];
	fill(num, num + rhs.nelements, lhs);
	vmdMul(rhs.nelements, num, rhs.content, result.content, mode);
	return result;
}
RealMatrix& RealMatrix::operator*=(const double& rhs)
{
	double* num = new double[nelements];
	fill(num, num + nelements, rhs);
	double* result = new double[nelements];
	vmdMul(nelements, num, content, result, mode);
	swap(content, result);
	delete[] result;
	return *this;
}
RealMatrix operator/(const RealMatrix& lhs, const double& rhs)
{
	return lhs * (1.0 / rhs);
}
RealMatrix& RealMatrix::operator/=(const double& rhs)
{
	*this *= 1.0 / rhs;
	return *this;
}

// assignment operator
RealMatrix& RealMatrix::operator=(const RealMatrix& rhs)
{
	if (length != rhs.length)
	{
		cerr << "ASSIGN DIFFERENT SIZE REAL MATRIX" << endl;
		exit(104);
	}
	copy(rhs.content, rhs.content + nelements, content);
	return *this;
}
RealMatrix& RealMatrix::operator=(const double* array)
{
	copy(array, array + nelements, content);
	return *this;
}


// ComplexMatrix functions
// default constructor with all zero
ComplexMatrix::ComplexMatrix(const int size)
	: length(size), nelements(length * length), content(new Complex[nelements])
{
	memset(content, 0, nelements * sizeof(Complex));
}

// copy constructor
ComplexMatrix::ComplexMatrix(const ComplexMatrix& matrix)
	: length(matrix.length), nelements(matrix.nelements), content(new Complex[nelements])
{
	copy(matrix.content, matrix.content + matrix.nelements, content);
}

// quasi copy constructor
ComplexMatrix::ComplexMatrix(const int size, const Complex* array)
	: length(size), nelements(length * length), content(new Complex[nelements])
{
	copy(array, array + nelements, content);
}

// copy constructor from real matrix
ComplexMatrix::ComplexMatrix(const RealMatrix& matrix)
	: length(matrix.length), nelements(length * length), content(new Complex[nelements])
{
	real_to_complex(matrix.content, content, nelements);
}

// quasi copy constructor from real matrix
ComplexMatrix::ComplexMatrix(const int size, const double* array)
	: length(size), nelements(length * length), content(new Complex[nelements])
{
	real_to_complex(array, content, nelements);
}

// move constructor
ComplexMatrix::ComplexMatrix(ComplexMatrix&& matrix)
	: length(move(matrix.length)), nelements(move(matrix.nelements)), content(move(matrix.content))
{
}


// one element is give number and the other are all zero
ComplexMatrix::ComplexMatrix(const int size, const Index& idx, const Complex& val)
	: length(size), nelements(length * length), content(new Complex[nelements])
{
	memset(content, 0, nelements * sizeof(Complex));
	content[idx.first * length + idx.second] = val;
}

// destructor
ComplexMatrix::~ComplexMatrix(void)
{
	delete[] content;
}

// the size of the matrix
int ComplexMatrix::length_of_matrix(void) const
{
	return length;
}

// direct access to internal data
Complex* ComplexMatrix::data(void)
{
	return content;
}
const Complex* ComplexMatrix::data(void) const
{
	return content;
}

// copy to an array
void ComplexMatrix::transform_to_1d(Complex* array) const
{
	copy(content, content + nelements, array);
}

// overload operator(): return the element (=[][])
Complex& ComplexMatrix::operator()(const int idx1, const int idx2)
{
	return content[idx1 * length + idx2];
}
Complex& ComplexMatrix::operator()(const Index& idx)
{
	return content[idx.first * length + idx.second];
}
const Complex& ComplexMatrix::operator()(const int idx1, const int idx2) const
{
	return content[idx1 * length + idx2];
}
const Complex& ComplexMatrix::operator()(const Index& idx) const
{
	return content[idx.first * length + idx.second];
}

// overload numerical calculation
ComplexMatrix operator+(const ComplexMatrix& lhs, const ComplexMatrix& rhs)
{
	if (lhs.length != rhs.length)
	{
		cerr << "ADD DIFFERENT SIZE RCOMPLEX MATRIX" << endl;
		exit(105);
	}
	ComplexMatrix result(lhs.length);
	vmzAdd(result.nelements, reinterpret_cast<const MKL_Complex16*>(lhs.content), reinterpret_cast<const MKL_Complex16*>(rhs.content), reinterpret_cast<MKL_Complex16*>(result.content), mode);
	return result;
}
ComplexMatrix& ComplexMatrix::operator+=(const ComplexMatrix& rhs)
{
	if (length != rhs.length)
	{
		cerr << "ADD-ASSIGN DIFFERENT SIZE COMPLEX MATRIX" << endl;
		exit(106);
	}
	Complex* result = new Complex[nelements];
	vmzAdd(nelements, reinterpret_cast<const MKL_Complex16*>(content), reinterpret_cast<const MKL_Complex16*>(rhs.content), reinterpret_cast<MKL_Complex16*>(result), mode);
	swap(content, result);
	delete[] result;
	return *this;
}
ComplexMatrix operator-(const ComplexMatrix& lhs, const ComplexMatrix& rhs)
{
	if (lhs.length != rhs.length)
	{
		cerr << "SUBTRACT DIFFERENT SIZE COMPLEX MATRIX" << endl;
		exit(107);
	}
	ComplexMatrix result(lhs.length);
	vmzSub(result.nelements, reinterpret_cast<const MKL_Complex16*>(lhs.content), reinterpret_cast<const MKL_Complex16*>(rhs.content), reinterpret_cast<MKL_Complex16*>(result.content), mode);
	return result;
}
ComplexMatrix& ComplexMatrix::operator-=(const ComplexMatrix& rhs)
{
	if (length != rhs.length)
	{
		cerr << "SUBTRACT-ASSIGN DIFFERENT SIZE COMPLEX MATRIX" << endl;
		exit(108);
	}
	Complex* result = new Complex[nelements];
	vmzSub(nelements, reinterpret_cast<const MKL_Complex16*>(content), reinterpret_cast<const MKL_Complex16*>(rhs.content), reinterpret_cast<MKL_Complex16*>(result), mode);
	swap(content, result);
	delete[] result;
	return *this;
}
ComplexMatrix operator*(const ComplexMatrix& lhs, const Complex& rhs)
{
	ComplexMatrix result(lhs.length);
	Complex* num = new Complex[lhs.nelements];
	fill(num, num + lhs.nelements, rhs);
	vmzMul(lhs.nelements, reinterpret_cast<const MKL_Complex16*>(lhs.content), reinterpret_cast<const MKL_Complex16*>(num), reinterpret_cast<MKL_Complex16*>(result.content), mode);
	return result;
}
ComplexMatrix operator*(const Complex& lhs, const ComplexMatrix& rhs)
{
	ComplexMatrix result(rhs.length);
	Complex* num = new Complex[rhs.nelements];
	fill(num, num + rhs.nelements, lhs);
	vmzMul(rhs.nelements, reinterpret_cast<const MKL_Complex16*>(num), reinterpret_cast<const MKL_Complex16*>(rhs.content), reinterpret_cast<MKL_Complex16*>(result.content), mode);
	return result;
}
ComplexMatrix& ComplexMatrix::operator*=(const Complex& rhs)
{
	Complex* num = new Complex[nelements];
	fill(num, num + nelements, rhs);
	Complex* result = new Complex[nelements];
	vmzMul(nelements, reinterpret_cast<const MKL_Complex16*>(num), reinterpret_cast<const MKL_Complex16*>(content), reinterpret_cast<MKL_Complex16*>(result), mode);
	swap(content, result);
	delete[] result;
	return *this;
}
ComplexMatrix operator/(const ComplexMatrix& lhs, const Complex& rhs)
{
	return lhs * (1.0 / rhs);
}
ComplexMatrix& ComplexMatrix::operator/=(const Complex& rhs)
{
	*this *= 1.0 / rhs;
	return *this;
}

// assignment operator
ComplexMatrix& ComplexMatrix::operator=(const ComplexMatrix& rhs)
{
	if (length != rhs.length)
	{
		cerr << "ASSIGN DIFFERENT SIZE COMPLEX MATRIX" << endl;
		exit(109);
	}
	copy(rhs.content, rhs.content + nelements, content);
	return *this;
}
ComplexMatrix& ComplexMatrix::operator=(const Complex* array)
{
	copy(array, array + nelements, content);
	return *this;
}
