/// @file general.h
/// @brief Declarations of variables and functions used in main driver: I/O, evolution, etc 
///
/// This header file contains some functions that will be used 
/// in main function, but they are often call once or twice.
/// However, putting the function codes directly in main make the
/// main function and main file too long. In that case, these 
/// functions are moved to an individual file (general.cpp)
/// and the interfaces are declared here.

#ifndef GENERAL_H
#define GENERAL_H

#include <iostream>
#include <memory>
#include <tuple>
#include "matrix.h"

const double pi = acos(-1.0); ///< mathematical constant, pi
const double hbar = 1.0; ///< physical constant, reduced Planck constant
const double PlanckH = 2 * pi * hbar; ///< physical constant, Planck constant
// Alpha and Beta are for cblas complex matrix multiplication functions
// as they behave as A=alpha*B*C+beta*A
const Complex Alpha(1.0, 0.0); ///< constant for cblas, A=alpha*B*C+beta*A
const Complex Beta(0.0, 0.0); ///< constant for cblas, A=alpha*B*C+beta*A
// for RK4, things are different
const Complex RK4kAlpha = 1.0 / 1.0i / hbar; ///< constant for RK4-calling cblas, k_{n+1}=H/ihbar*(psi+dt/1~2*kn)

const double PplLim = 1e-4; ///< if the overall population on all PES is smaller than PplLim, it is stable and could stop simulation. Used only with ABC
const double ChangeLim = 1e-5; ///< if the population change on each PES is smaller than ChangeLim, then it is stable and could stop simulation


// utility functions

/// @brief cut off
/// @param val the input value to do the cutoff
/// @return the value after cutoff
double cutoff(const double val);

/// sign function; return -1 for negative, 1 for positive, and 0 for 0
/// @param val a value of any type that have '<' and '>' and could construct 0.0
/// @return the sign of the value
template <typename valtype>
inline int sgn(const valtype& val)
{
    return (val > valtype(0.0)) - (val < valtype(0.0));
}

/// @brief returns (-1)^n
/// @param n an integer
/// @return -1 if n is odd, or 1 if n is even
int pow_minus_one(const int n);


// I/O functions

/// @brief read a double: mass, x0, etc
/// @param is an istream object (could be ifstream/isstream) to input
/// @return the real value read from the stream
double read_double(istream& is);

/// @brief check if absorbing potential is used or not
/// @param is an istream object (could be ifstream/isstream) to input
/// @return a boolean, true means Absorbed Boundary Condition (ABC) is used, false means not
bool read_absorb(istream& is);

/// @brief to print current time
/// @param os an ostream object (could be ifstream/isstream) to output
/// @return the ostream object of the parameter after output the time
ostream& show_time(ostream& os);


// evolution related functions

/// @brief construct the diabatic Hamiltonian
/// @param NGrids the number of grids in wavefunction
/// @param GridCoordinate the coordinate of each grid, i.e., x_i
/// @param dx the grid spacing
/// @param mass the mass of the bath (the nucleus mass)
/// @param absorbed whethe the ABC is used or not
/// @param xmin the left boundary, only used with ABC
/// @param xmax the right boundary, only used with ABC
/// @param AbsorbingRegionLength the extended region for absoptiong, only used with ABC
/// @return the Hamiltonian, a complex matrix, hermitian if no ABC
ComplexMatrix Hamiltonian_construction
(
    const int NGrids,
    const double* const GridCoordinate,
    const double dx,
    const double mass,
    const bool Absorbed,
    const double xmin,
    const double xmax,
    const double AbsorbingRegionLength
);

/// @brief initialize the gaussian wavepacket, and normalize it
/// @param NGrids the number of grids in wavefunction
/// @param GridCoordinate the coordinate of each grid, i.e., x_i
/// @param dx the grid spacing
/// @param x0 the initial average position
/// @param p0 the initial average momentum
/// @param SigmaX the initial standard deviation of position. SigmaP is not needed due to the minimum uncertainty principle
/// @param psi the wavefunction to be initialized in adiabatic representation
void wavefunction_initialization(
    const int NGrids,
    const double* const GridCoordinate,
    const double dx,
    const double x0,
    const double p0,
    const double SigmaX,
    Complex* const Psi
);

/// @brief calculate the phase space distribution, and output it
/// @param os an ostream object (could be ofstream/osstream) to output
/// @param NGrids the number of grids in wavefunction
/// @param GridCoordinate the coordinate of each grid, i.e., x_i
/// @param dx the grid spacing
/// @param p0 the initial average momentum
/// @param psi the wavefunction
void output_phase_space_distribution
(
    ostream& os,
    const int NGrids,
    const double* const GridCoordinate,
    const double dx,
    const double p0,
    const Complex* const Psi
);

/// @brief calculate the population on each PES
/// @param NGrids the number of grids in wavefunction
/// @param dx the grid spacing
/// @param psi the wavefunction
/// @param Population the array to save the population on each PES
void calculate_popultion
(
    const int NGrids,
    const double dx,
    const Complex* const psi,
    double* const Population
);

#endif // !GENERAL_H