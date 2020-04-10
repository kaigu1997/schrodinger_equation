/// @file general.cpp
/// @brief Implementation of general.h

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <iostream>
#include <memory>
#include <mkl.h>
#include <string>
#include <tuple>
#include <utility>
#include "general.h"
#include "matrix.h"
#include "pes.h"
using namespace std;

// utility functions

/// Do the cutoff, e.g. 0.2493 -> 0.125, 1.5364 -> 1
/// 
/// Transform to the nearest 2 power, i.e. 2^(-2), 2^0, etc
double cutoff(const double val)
{
    return exp2(static_cast<int>(floor(log2(val))));
}

int pow_minus_one(const int n)
{
    return n % 2 == 0 ? 1 : -1;
}



// I/O functions

/// The file is one line declarator and one line the value,
/// so using a buffer string to read the declarator
/// and the rest of the second line.
double read_double(istream& is)
{
    static string buffer;
    static double temp;
    getline(is, buffer);
    is >> temp;
    getline(is, buffer);
    return temp;
}

/// Judge by the string in the input.
///
/// If the string is "on", use ABC, return true.
///
/// If the string is "off", not use ABC, return false.
///
/// Otherwise, the input is invalid, kills the program
bool read_absorb(istream& is)
{
    string buffer, WhetherAbsorb;
    bool Absorbed;
    getline(is, buffer);
    is >> WhetherAbsorb;
    getline(is, buffer);
    if (strcmp(WhetherAbsorb.c_str(), "on") == 0)
    {
        Absorbed = true;
    }
    else if (strcmp(WhetherAbsorb.c_str(), "off") == 0)
    {
        Absorbed = false;
    }
    else
    {
        cerr << "UNKNOWN CASE OF ABSORBING POTENTIAL" << endl;
        exit(301);
    }
    return Absorbed;
}

ostream& show_time(ostream& os)
{
    auto time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    os << ctime(&time);
    return os;
}


// evolution related function
/// Construct the diabatic Hamiltonian consisting of 3 parts:
///
/// 1. diabatic potential (the subsystem/electron Hamiltonian), which is real-symmetric
///
/// 2. bath/nucleus kinetic energy by infinite order finite difference, which is also real-symmetric
///
/// 3. absorbed boundary condition when it is used, which is pure complex on diagonal elements
///
/// This function is called only once for the initialization of Hamiltonian
ComplexMatrix Hamiltonian_construction
(
    const int NGrids,
    const double* const GridCoordinate,
    const double dx,
    const double mass,
    const bool absorbed,
    const double xmin,
    const double xmax,
    const double AbsorbingRegionLength
)
{
    ComplexMatrix Hamiltonian(NGrids * NumPES);
    // 1. V_{mm'}(R_n), n=n'
    for (int n = 0; n < NGrids; n++)
    {
        const RealMatrix& Vn = diabatic_potential(GridCoordinate[n]);
        for (int m = 0; m < NumPES; m++)
        {
            for (int mm = 0; mm < NumPES; mm++)
            {
                Hamiltonian[m * NGrids + n][mm * NGrids + n] += Vn[m][mm];
            }
        }
    }
    // 2. d2/dx2 (over all pes)
    for (int m = 0; m < NumPES; m++)
    {
        for (int n = 0; n < NGrids; n++)
        {
            for (int nn = 0; nn < NGrids; nn++)
            {
                if (nn == n)
                {
                    Hamiltonian[m * NGrids + n][m * NGrids + nn] += pow(pi * hbar / dx, 2) / 6.0 / mass;
                }
                else
                {
                    Hamiltonian[m * NGrids + n][m * NGrids + nn] += pow_minus_one(nn - n) * pow(hbar / dx / (nn - n), 2) / mass;
                }
            }
        }
    }
    // 3. absorbing potential
    if (absorbed == true)
    {
        for (int n = 0; n < NGrids; n++)
        {
            const double&& An = absorb_potential(mass, xmin, xmax, AbsorbingRegionLength, GridCoordinate[n]);
            for (int m = 0; m < NumPES; m++)
            {
                Hamiltonian[m * NGrids + n][m * NGrids + n] -= 1.0i * An;
            }
        }
    }
    return Hamiltonian;
}

/// psi(x)=exp(-(x-x0)^2/4/sigma_x^2+i*p0*x/hbar)/[(2*pi)^(1/4)*sqrt(sigma_x)]
///
/// as the wavefunction is on grid, the normalization factor could be different
///
/// this function is also only called once, for initialization of the adiabatic wavefunction
void wavefunction_initialization(const int NGrids, const double* GridCoordinate, const double dx, const double x0, const double p0, const double SigmaX, Complex* psi)
{
    // for higher PES, the initial wavefunction is zero
    memset(psi + NGrids, 0, NGrids * (NumPES - 1) * sizeof(Complex));
    // for ground state, it is a gaussian. psi(x)=A*exp(-(x-x0)^2/4sigmax^2+ip0x/hbar)
    for (int i = 0; i < NGrids; i++)
    {
        const double& x = GridCoordinate[i];
        psi[i] = exp(-pow((x - x0) / 2 / SigmaX, 2) + p0 * x / hbar * 1.0i) / sqrt(sqrt(2.0 * pi) * SigmaX);
    }
    // normalization
    Complex PsiSquare;
    // calling cblas_zdotc_sub rather than cblas_znrm2 due to higher accuracy
    cblas_zdotc_sub
    (
        NGrids,
        psi,
        1,
        psi,
        1,
        &PsiSquare
    );
    const double NormFactor = sqrt(PsiSquare.real() * dx);
    for (int i = 0; i < NGrids; i++)
    {
        psi[i] /= NormFactor;
    }
}

/// Output one element of ComplexMatrix in a line, change column first, then row
///
/// That is to say,
///
/// (line 1) rho00(x0,p0) rho00(x0,p1) rho00(x0,p2) ... rho00(x0,pn) ... rho00(xn,pn)
///
/// (line 2) rho01(x0,p0) rho01(x0,p1) rho01(x0,p2) ... rho01(x0,pn) ... rho01(xn,pn)
///
/// and so on, and a blank line at the end.
///
/// The region of momentum is the Fourier transformation of position, and moved to centered at initial momentum.
///
/// In other words, p in p0+pi*hbar/dx*[-1,1], dp=2*pi*hbar/(xmax-xmin)
void output_phase_space_distribution
(
    ostream& os,
    const int NGrids,
    const double* GridCoordinate,
    const double dx,
    const double p0,
    const Complex* psi
)
{
    const double pmin = p0 - pi * hbar / dx;
    const double pmax = p0 + pi * hbar / dx;
    // Wigner Transformation: P(x,p)=int{dy*exp(2ipy/hbar)<x-y|rho|x+y>}/(pi*hbar)
    // the interval of p is p0+pi*hbar/dx*[-1,1), dp=2*pi*hbar/(xmax-xmin)
    // loop over pes first
    for (int iPES = 0; iPES < NumPES; iPES++)
    {
        for (int jPES = 0; jPES < NumPES; jPES++)
        {
            // loop over x
            for (int xi = 0; xi < NGrids; xi++)
            {
                // loop over p
                for (int pj = 0; pj < NGrids; pj++)
                {
                    const double p = ((NGrids - 1 - pj) * pmin + pj * pmax) / (NGrids - 1);
                    // do the numerical integral
                    Complex integral(0.0, 0.0);
                    // 0 <= (x+y, x-y) < NGrids 
                    for (int yk = max(-xi, xi + 1 - NGrids); yk <= min(xi, NGrids - 1 - xi); yk++)
                    {
                        const double y = GridCoordinate[0] + yk * dx;
                        integral += exp(2.0 * p * y / hbar * 1.0i) * psi[xi - yk + iPES * NGrids] * conj(psi[xi + yk + jPES * NGrids]);
                    }
                    integral *= dx / pi / hbar;
                    os << ' ' << integral.real() << ' ' << integral.imag();
                }
            }
            os << '\n';
        }
    }
    os << endl;
}

// calculate the population on each PES
void calculate_popultion
(
    const int NGrids,
    const double dx,
    const Complex* AdiabaticPsi,
    double* Population
)
{
    Complex InnerProduct;
    // calculate the inner product of each PES
    // again, calling cblas_zdotc_sub rather than cblas_znrm2 due to higher accuracy
    for (int i = 0; i < NumPES; i++)
    {
        cblas_zdotc_sub
        (
            NGrids,
            AdiabaticPsi + i * NGrids,
            1,
            AdiabaticPsi + i * NGrids,
            1,
            &InnerProduct
        );
        Population[i] = InnerProduct.real() * dx;
    }
}
