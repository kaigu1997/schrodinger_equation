#ifndef PES_H
#define PES_H

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
#include "general.h"
#include "matrix.h"
using namespace std;

// the number of potential energy surfaces
const int NumPES = 2;
// diabatic PES matrix: analytical
RealMatrix DiaPotential(const double x);
// the absorbing potential: V->V-iE
// Here is E only. E is diagonal
double AbsorbPotential(const double mass, const double xmin, const double xmax, const double AbsorbingRegionLength, const double x);

// transformation matrix from diabatic state to adiabatic state
// i.e. M*psi(dia)=psi(adia), which diagonalize PES only (instead of diagonal H)
ComplexMatrix DiaToAdia(const int NGrids, const double* GridCoordinate);

#endif // !PES_H
