// implementation of general.h

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

// returns (-1)^n
int pow_minus_one(const int n)
{
    return n % 2 == 0 ? 1 : -1;
}

// do the cutoff, e.g. 0.2493 -> 0.2, 1.5364 -> 1
double cutoff(const double val)
{
    double pownum = pow(10, static_cast<int>(floor(log10(val))));
    return static_cast<int>(val / pownum) * pownum;
}


// I/O functions

// read a double: mass, x0, etc
double read_double(istream& is)
{
    static string buffer;
    static double temp;
    getline(is, buffer);
    is >> temp;
    getline(is, buffer);
    return temp;
}

// check if absorbing potential is used or not
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

// to print current time
ostream& show_time(ostream& os)
{
    auto time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    os << ctime(&time);
    return os;
}


// evolution related function

// initialize the gaussian wavepacket, and normalize it
ComplexVector wavefunction_initialization(const int NGrids, const double* GridCoordinate, const double dx, const double x0, const double p0, const double SigmaX)
{
    // generate the array to save the initial wavepacket
    Complex* Psi0 = new Complex[NGrids * NumPES];
    // for higher PES, the initial wavefunction is zero
    memset(Psi0 + NGrids, 0, NGrids * (NumPES - 1) * sizeof(Complex));
    // for ground state, it is a gaussian. psi(x)=A*exp(-(x-x0)^2/4sigmax^2+ip0x/hbar)
    for (int i = 0; i < NGrids; i++)
    {
        Psi0[i] = exp(Complex(-pow((GridCoordinate[i] - x0) / 2 / SigmaX, 2), p0 * GridCoordinate[i] / hbar));
    }
    // normalization
    Complex PsiSquare;
    cblas_zdotc_sub(NGrids, Psi0, 1, Psi0, 1, &PsiSquare);
    double NormFactor = sqrt(PsiSquare.real() * dx);
    for (int i = 0; i < NGrids; i++)
    {
        Psi0[i] /= NormFactor;
    }
    // give it to a unique_ptr
    return ComplexVector(Psi0);
}

// construct the Hamiltonian
ComplexMatrix Hamiltonian_construction(const int NGrids, const double* GridCoordinate, const double dx, const double mass, const bool Absorbed, const double xmin, const double xmax, const double AbsorbingRegionLength)
{
    ComplexMatrix Hamiltonian(NGrids * NumPES);
    // 1. V_{mm'}(R_n), n=n'
    for (int n = 0; n < NGrids; n++)
    {
        const RealMatrix&& Vn = DiaPotential(GridCoordinate[n]);
        for (int m = 0; m < NumPES; m++)
        {
            for (int mm = 0; mm < NumPES; mm++)
            {
                Hamiltonian(m * NGrids + n, mm * NGrids + n) += Vn(m, mm);
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
                    Hamiltonian(m * NGrids + n, m * NGrids + nn) += pow(pi * hbar / dx, 2) / 6.0 / mass;
                }
                else
                {
                    Hamiltonian(m * NGrids + n, m * NGrids + nn) += pow_minus_one(nn - n) * pow(hbar / dx / (nn - n), 2) / mass;
                }
            }
        }
    }
    // add absorbing potential
    if (Absorbed == true)
    {
        for (int n = 0; n < NGrids; n++)
        {
            const double&& An = AbsorbPotential(mass, xmin, xmax, AbsorbingRegionLength, GridCoordinate[n]);
            for (int m = 0; m < NumPES; m++)
            {
                Hamiltonian(m * NGrids + n, m * NGrids + n) -= 1.0i * An;
            }
        }
    }
    return Hamiltonian;
}

// calculate the population on each PES
void calculate_popultion(const int NGrids, const double dx, const Complex* AdiabaticPsi, double* Population)
{
    static Complex InnerProduct;
    // calculate the inner product of each PES
    for (int i = 0; i < NumPES; i++)
    {
        cblas_zdotc_sub(NGrids, AdiabaticPsi + i * NGrids, 1, AdiabaticPsi + i * NGrids, 1, &InnerProduct);
        Population[i] = InnerProduct.real() * dx;
    }
}
