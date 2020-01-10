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
#include <functional>
#include <mkl.h>
#include <utility>
#include "matrix.h"
using namespace std;

// mathematical, physical and computational constants
const double pi = acos(-1.0), hbar = 1.0, PlanckH = 2 * pi * hbar;
const Complex Alpha(1.0, 0.0), Beta(0.0, 0.0);
// the representation
enum Representation {Diabatic, Adiabatic, Force};

// the number of potential energy surfaces
const int NumPES = 2;
// diabatic PES matrix: analytical
RealMatrix DiaPotential(const double x);
// adiabatic PES: diagonal matrix
RealMatrix AdiaPotential(const double x);
// nonadiabatic coupling under adiabatic representation
RealMatrix AdiaNAC(const double x);
// force basis PES
RealMatrix ForcePotential(const double x);
// nonadiabatic coupling under force basis
RealMatrix ForceNAC(const double x);
// saves in function pointer array
const function<RealMatrix(const double)> PES[3] = { DiaPotential, AdiaPotential, ForcePotential }, NAC[3] = { nullptr, AdiaNAC, ForceNAC };
// the absorbing potential: V->V-iE
// Here is E only. E is diagonal
double AbsorbPotential(const double mass, const double xmin, const double xmax, const double AbsorbingRegionLength, const double x);
// basis transformation and saves in function pointer array
void DiaToAdia(const int NGrids, double* GridCoordinate, Complex* Psi);
void DiaToForce(const int NGrids, double* GridCoordinate, Complex* Psi);
void AdiaToDia(const int NGrids, double* GridCoordinate, Complex* Psi);
void AdiaToForce(const int NGrids, double* GridCoordinate, Complex* Psi);
void ForceToDia(const int NGrids, double* GridCoordinate, Complex* Psi);
void ForceToAdia(const int NGrids, double* GridCoordinate, Complex* Psi);
const function<void(const int,double*,Complex*)> PsiTransformation[3][3] =
{
    {nullptr, DiaToAdia, DiaToForce},
    {AdiaToDia, nullptr, AdiaToForce},
    {ForceToDia, ForceToAdia, nullptr}
};

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
