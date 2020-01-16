#ifndef GENERAL_H
#define GENERAL_H

// this header file contains the
// parameter used in DVR calculation
// and gives the Potential Energy Surface
// (PES) of different representations:
// diabatic, adiabatic and force-basis.
// Transformation is done by unitary
// matrices calculate by LAPACK in MKL

#include <algorithm>
#include <cmath>
#include <complex>
#include <mkl.h>
#include <utility>
#include "matrix.h"
using namespace std;

// mathematical, physical and computational constants
const double pi = acos(-1.0), hbar = 1.0, PlanckH = 2 * pi * hbar;
const Complex Alpha(1.0, 0.0), Beta(0.0, 0.0);

// the number of potential energy surfaces
const int NumPES = 2;
// diabatic PES matrix: analytical
RealMatrix DiaPotential(const double x);
// the absorbing potential: V->V-iE
// Here is E only. E is diagonal
double AbsorbPotential(const double mass, const double xmin, const double xmax, const double AbsorbingRegionLength, const double x);

// maximum number of grid, to prevent stackoverflow
// if NGrids > MaxNGrids, no change of dt
const int MaxNGrids = 301;

// sign function; working for all type that have '<' and '>'
// return -1 for negative, 1 for positive, and 0 for 0
template <typename valtype>
inline int sgn(const valtype& val)
{
    return (val > valtype(0.0)) - (val < valtype(0.0));
}

// returns (-1)^n
inline int pow_minus_one(const int n)
{
    return n % 2 == 0 ? 1 : -1;
}

#endif // !GENERAL_H
