#ifndef GENERAL_H
#define GENERAL_H

// This header file contains some functions that will be used 
// in main function, but they are often call once or twice.
// However, putting the function codes directly in main make the
// main function and main file too long. In that case, these 
// functions are moved to an individual file (general.cpp)
// and the interfaces are declared here.

#include <iostream>
#include <memory>
#include <tuple>
#include "matrix.h"

// mathematical and physical constants
const double pi = acos(-1.0), hbar = 1.0, PlanckH = 2 * pi * hbar;
// Alpha and Beta are for cblas complex matrix multiplication functions
// as they behave as A=alpha*B*C+beta*A
const Complex Alpha(1.0, 0.0), Beta(0.0, 0.0);
// for RK4, things are different
const Complex RK4kAlpha = 1.0 / 1.0i / hbar;
// PplLim is that, if the overall population on all PES is smaller 
// than PplLim, then it is stable and could stop simulation
const double PplLim = 1e-4;
// ChangeLim is that, if the population change on each PES is smaller 
// than ChangeLim, then it is stable and could stop simulation
const double ChangeLim = 1e-5;

// two kinds of unique_ptr array, no need to free manually
typedef unique_ptr<double[]> doubleVector;
typedef unique_ptr<Complex[]> ComplexVector;


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

// check if absorbing potential is used or not
bool read_absorb(istream& is);

// to print current time
ostream& show_time(ostream& os);


// evolution related functions

// construct the Hamiltonian
ComplexMatrix Hamiltonian_construction(const int NGrids, const double* GridCoordinate, const double dx, const double mass, const bool Absorbed, const double xmin, const double xmax, const double AbsorbingRegionLength);

// initialize the gaussian wavepacket, and normalize it
void wavefunction_initialization(const int NGrids, const double* GridCoordinate, const double dx, const double x0, const double p0, const double SigmaX, Complex* psi);

// calculate the phase space distribution, and output it
void output_phase_space_distribution(ostream& out, const int NGrids, const double* GridCoordinate, const double dx, const Complex* psi);


// calculate the population on each PES
void calculate_popultion(const int NGrids, const double dx, const Complex* AdiabaticPsi, double* Population);

#endif // !GENERAL_H