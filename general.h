#ifndef GENERAL_H
#define GENERAL_H

// This header file contains some functions that will be used 
// in main function, but they are often call once or twice.
// However, putting the function codes directly in main make the
// main function and main file too long. In that case, these 
// functions are moved to an individual file (general.cpp)
// and the interfaces are declared here.

#include <iostream>
#include "matrix.h"

// mathematical and physical constants
const double pi = acos(-1.0), hbar = 1.0, PlanckH = 2 * pi * hbar;
// Alpha and Beta are for cblas complex matrix multiplication functions
// as they behave as A=alpha*B*C+beta*A
const Complex Alpha(1.0, 0.0), Beta(0.0, 0.0);
// for RK4, things are different
const Complex RK4kAlpha = 1.0 / 1.0i / hbar;
// AbsorbLim means if the absorbed population is over this
// then the program should stop. Used only when Absorb is on
const double AbsorbLim = 1.0e-2;
// PplLim is that, if the overall population on all PES is smaller 
// than PplLim, then it is stable and could stop simulation
const double PplLim = 1e-4;


// utility functions

// do the cutoff, e.g. 0.2493 -> 0.2, 1.5364 -> 1
double cutoff(const double val);

// sign function; working for all type that have '<' and '>'
// return -1 for negative, 1 for positive, and 0 for 0
template <typename valtype>
int sgn(const valtype& val)
{
    return (val > valtype(0.0)) - (val < valtype(0.0));
}

// returns (-1)^n
int pow_minus_one(const int n);


// I/O functions

// read a double: mass, x0, etc
double read_double(istream& is);

// to print current time
ostream& show_time(ostream& os);


// evolution related functions

// construct the Hamiltonian
ComplexMatrix Hamiltonian_construction(const int NGrids, const double* GridCoordinate, const double dx, const double mass);

// initialize the gaussian wavepacket, and normalize it
void diabatic_wavefunction_initialization(const int NGrids, const double* GridCoordinate, const double dx, const double x0, const double p0, const double SigmaX, Complex* psi);

// cutoff the out-of-boundary populations
void absorb(const int NGrids, const int LeftIndex, const int RightIndex, const double dx, Complex* psi, double* AbsorbedPopulation);

// calculate the population on each PES
void calculate_popultion(const int NGrids, const double dx, const Complex* AdiabaticPsi, double* Population);

#endif // !GENERAL_H