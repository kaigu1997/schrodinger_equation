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

// to print current time
ostream& show_time(ostream& os)
{
    auto time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    os << ctime(&time);
    return os;
}


// evolution related function

// construct the Hamiltonian
ComplexMatrix Hamiltonian_construction(const int NGrids, const double* GridCoordinate, const double dx, const double mass)
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
    return Hamiltonian;
}

// initialize the gaussian wavepacket, and normalize it
void diabatic_wavefunction_initialization(const int NGrids, const double* GridCoordinate, const double dx, const double x0, const double p0, const double SigmaX, Complex* psi)
{
    // for higher PES, the initial wavefunction is zero
    memset(psi + NGrids, 0, NGrids * (NumPES - 1) * sizeof(Complex));
    // for ground state, it is a gaussian. psi(x)=A*exp(-(x-x0)^2/4sigmax^2+ip0x/hbar)
    for (int i = 0; i < NGrids; i++)
    {
        psi[i] = exp(Complex(-pow((GridCoordinate[i] - x0) / 2 / SigmaX, 2), p0 * GridCoordinate[i] / hbar));
    }
    // normalization
    Complex PsiSquare;
    cblas_zdotc_sub(NGrids, psi, 1, psi, 1, &PsiSquare);
    double NormFactor = sqrt(PsiSquare.real() * dx);
    for (int i = 0; i < NGrids; i++)
    {
        psi[i] /= NormFactor;
    }
}

// cutoff the out-of-boundary populations
void absorb(const int NGrids, const int LeftIndex, const int RightIndex, const double dx, Complex* psi, double* AbsorbedPopulation)
{
    for (int i = 0; i < NumPES; i++)
    {
        // absorb the left
        for (int j = 0; j < LeftIndex; j++)
        {
            AbsorbedPopulation[i] += (psi[j] * conj(psi[j])).real() * dx;
            psi[j] = 0;
        }
        // absorb the right
        for (int j = RightIndex + 1; j < NGrids; j++)
        {
            AbsorbedPopulation[i + NumPES] += (psi[j] * conj(psi[j])).real() * dx;
            psi[j] = 0;
        }
    }
}

// calculate the population on each PES
void calculate_popultion(const int NGrids, const double dx, const Complex* AdiabaticPsi, double* Population)
{
    Complex InnerProduct;
    // calculate the inner product of each PES
    for (int i = 0; i < NumPES; i++)
    {
        cblas_zdotc_sub(NGrids, AdiabaticPsi + i * NGrids, 1, AdiabaticPsi + i * NGrids, 1, &InnerProduct);
        Population[i] = InnerProduct.real() * dx;
    }
}
