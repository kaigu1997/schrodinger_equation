// implementation of general.h:
// diabatic PES and Hamiltonians

#include <algorithm>
#include <cmath>
#include <iostream>
#include <mkl.h>
#include "general.h"
#include "matrix.h"

/*Model 1
// constants in the diabatic Potential
static const double A = 0.01, B = 1.6, C = 0.005, D = 1.0;
// subsystem diabatic Potential, the force, and hessian
// the "forces" are negative derivative over x
// the hessian matrix is the second derivative over x
RealMatrix DiaPotential(const double x)
{
    RealMatrix Potential(NumPES);
    Potential(0, 1) = Potential(1, 0) = C * exp(-D * x * x);
    Potential(0, 0) = sgn(x) * A * (1.0 - exp(-sgn(x) * B * x));
    Potential(1, 1) = -Potential(0, 0);
    return Potential;
}
static RealMatrix DiaForce(const double x)
{
    RealMatrix Force(NumPES);
    Force(0, 1) = Force(1, 0) = 2.0 * C * D * x * exp(-D * x * x);
    Force(0, 0) = -A * B * exp(-sgn(x) * B * x);
    Force(1, 1) = -Force[0][0];
    return Force;
}
static RealMatrix DiaHesse(const double x)
{
    RealMatrix Hesse(NumPES);
    Hesse(0, 1) = Hesse(1, 0) = 2 * C * D * (2 * D * x * x - 1) * exp(-D * x * x);
    Hesse(0, 0) = -sgn(x) * A * B * B * exp(-sgn(x) * B * x);
    Hesse(1, 1) = -Hesse(0, 0);
    return Hesse;
}// */

//*Model 2
// constants in the diabatic Potential
static const double A = 0.10, B = 0.28, C = 0.015, D = 0.06, E = 0.05;
// subsystem diabatic Potential, the force, and hessian
// the "forces" are negative derivative over x
// the hessian matrix is the second derivative over x
RealMatrix DiaPotential(const double x)
{
    RealMatrix Potential(NumPES);
    Potential(0, 1) = Potential(1, 0) = C * exp(-D * x * x);
    Potential(1, 1) = E - A * exp(-B * x * x);
    return Potential;
}
static RealMatrix DiaForce(const double x)
{
    RealMatrix Force(NumPES);
    Force(0, 1) = Force(1, 0) = 2 * C * D * x * exp(-D * x * x);
    Force(1, 1) = -2 * A * B * x * exp(-B * x * x);
    return Force;
}
static RealMatrix DiaHesse(const double x)
{
    RealMatrix Hesse(NumPES);
    Hesse(0, 1) = Hesse(1, 0) = 2 * C * D * (2 * D * x * x - 1) * exp(-D * x * x);
    Hesse(1, 1) = -2 * A * B * (2 * B * x * x - 1) * exp(-B * x * x);
    return Hesse;
}// */

/*Model 3
// constants in the diabatic Potential
static const double A = 6e-4, B = 0.10, C = 0.90;
// subsystem diabatic Potential, the force, and hessian
// the "forces" are negative derivative over x
// the hessian matrix is the second derivative over x
RealMatrix DiaPotential(const double x)
{
    RealMatrix Potential(NumPES);
    Potential(0, 0) = A;
    Potential(1, 1) = -A;
    Potential(0, 1) = Potential(1, 0) = B * (1 - sgn(x) * (exp(-sgn(x) * C * x) - 1));
    return Potential;
}
static RealMatrix DiaForce(const double x)
{
    RealMatrix Force(NumPES);
    Force(0, 1) = Force(1, 0) = -B * C * exp(-sgn(x) * C * x);
    return Force;
}
static RealMatrix DiaHesse(const double x)
{
    RealMatrix Hesse(NumPES);
    Hesse(0, 1) = Hesse(1, 0) = -sgn(x) * B * C * C * exp(-sgn(x) * C * x);
    return Hesse;
}// */


// Adiabatic Basis
// basis transformation matrix, transform
// from diabatic basis to adiabatic basis;
// using inner product to keep continuity
static RealMatrix DiaToAdiaMatrix(const double x)
{
    // EigVec is diabatic Hamiltonian before diag
    // and is the columnwise eigenvectors after diags
    RealMatrix EigVec(DiaPotential(x));
    // do the diagonalization to get the transformation matrix
    // EigVal is the eigenvalue of Hamiltonian (energy), only for storage
    double EigVal[NumPES];
    // diagonalize using d(ouble)sy(mmetric)e(igen)v(alue/ector) in LAPACK
    // and also calculate the columnwised eigenvector (adiabatic state) matrix
    if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', NumPES, EigVec.data(), NumPES, EigVal) > 0)
    {
        cerr << "FAILING CALCULATING ADIABATIC BASIS AT " << x << endl;
        exit(200);
    }
    return EigVec;
}

// adiabatic energy; the eigenvalues of Hamiltonian
// as diagonalization is required and the eigenvalues
// are the same, do diagonalization directly 
RealMatrix AdiaPotential(const double x)
{
    // EigVal is the eigenvalue of Hamiltonian (energy)
    double EigVal[NumPES];
    // diagonalize using d(ouble)sy(mmetric)e(igen)v(alue/ector) in LAPACK
    if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', NumPES, DiaPotential(x).data(), NumPES, EigVal) > 0)
    {
        cerr << "FAILING CALCULATING ADIABATIC HAMILTONIAN AT " << x << endl;
        exit(201);
    }
    // return the result
    RealMatrix Hamil(NumPES);
    for (int i = 0; i < NumPES; i++)
    {
        Hamil(i, i) = EigVal[i];
    }
    return Hamil;
}

// adiabatic forces; by transform from diabatic basis
static RealMatrix AdiaForce(const double x)
{
    // EigVec is diabatic Hamiltonian resized into 1D form before diag
    // and is the columnwise eigenvectors after diag
    // DiaF1D is diabatic "force" resized into 1D form
    const RealMatrix&& EigVec = DiaToAdiaMatrix(x);
    // FoEig is the matrix, [force] * [eigvec](column mat)
    double FoEig[NumPES * NumPES];
    // Force is the result matrix, [eigvec]^T * [force] * [eigvec]
    RealMatrix Force(NumPES);
    // using d(ouble)sy(mmetic)/ge(neral)m(atrix)m(ultiplication) in CBLAS to calculate the above
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, 1.0, DiaForce(x).data(), NumPES, EigVec.data(), NumPES, 0.0, FoEig, NumPES);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, NumPES, NumPES, NumPES, 1.0, EigVec.data(), NumPES, FoEig, NumPES, 0.0, Force.data(), NumPES);
    // after multiplication there might be numerical error
    // since F is hermician, average the off-diagonal
    for (int i = 0; i < NumPES; i++)
    {
        for (int j = i + 1; j < NumPES; j++)
        {
            Force(i, j) = Force(j, i) = (Force(i, j) + Force(j, i)) / 2.0;
        }
    }
    return Force;
}

// non-adiabatic coupling of adiabatic state
RealMatrix AdiaNAC(const double x)
{
    // NAC is the non-adiabatic coupling matrix, 0 for diag
    // AdiaF is the adiabatic "force"
    // AdiaH is the diagonal adiabatic hamiltonian
    RealMatrix NAC(NumPES);
    const RealMatrix&& AdiaF(AdiaForce(x)), AdiaH(AdiaPotential(x));
    // dij=Fij/(Ei-Ej)
    for (int i = 0; i < NumPES; i++)
    {
        for (int j = i + 1; j < NumPES; j++)
        {
            NAC(i, j) = AdiaF(i, j) / (AdiaH(i, i) - AdiaH(j, j));
            NAC(j, i) = -NAC(i, j);
        }
    }
    return NAC;
}


// Force Basis 
// basis transformation matrix, transform
// from diabatic basis to force basis;
// using inner product to keep continuity
// under such basis, the "force" (-dH/dR) is diagonalized
static RealMatrix DiaToForceMatrix(const double x)
{
    // EigVec is diabatic "Forces" before diag
    // and is the columnwise eigenvectors after diag
    RealMatrix EigVec(DiaForce(x));
    // EigVal is the eigenvalue of "Forces" (energy), only for storage
    double EigVal[NumPES];
    // diagonalize using d(ouble)sy(mmetric)e(igen)v(alue/ector) in LAPACK
    // and also calculate the columnwised eigenvector (adiabatic state) matrix
    if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', NumPES, EigVec.data(), NumPES, EigVal) > 0)
    {
        cerr << "FAILING CALCULATING FORCE BASIS AT " << x << endl;
        exit(202);
    }
    return EigVec;
}

// force basis; the "force" (-dH/dR) is diagonal
static RealMatrix ForceForce(const double x)
{
    // EigVal is the eigenvalue of "forces"
    double EigVal[NumPES];
    // diagonalize using d(ouble)sy(mmetric)e(igen)v(alue/ector) in LAPACK
    if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', NumPES, DiaForce(x).data(), NumPES, EigVal) > 0)
    {
        cerr << "FAILING CALCULATING DIAGONAL \"FORCE\" AT " << x << endl;
        exit(203);
    }
    // return the result
    RealMatrix Force(NumPES);
    for (int i = 0; i < NumPES; i++)
    {
        Force(i, i) = EigVal[i];
    }
    return Force;
}

// Hamiltonian under force basis
RealMatrix ForcePotential(const double x)
{
    // EigVec is diabatic "force" resized into 1D form before diag
    // and is the columnwise eigenvectors after diag
    // DiaH1D is diabatic Hamiltonian resized into 1D form
    const RealMatrix&& EigVec = DiaToForceMatrix(x);
    // HaEig is the matrix, [Hamil] * [eigvec](column mat)
    double HaEig[NumPES * NumPES];
    // Hamil is the result matrix, [eigvec]^T * [Hamil] * [eigvec]
    RealMatrix Hamil(NumPES);
    // using d(ouble)sy(mmetic)/ge(neral)m(atrix)m(ultiplication) in CBLAS to calculate the above
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, 1.0, DiaPotential(x).data(), NumPES, EigVec.data(), NumPES, 0.0, HaEig, NumPES);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, NumPES, NumPES, NumPES, 1.0, EigVec.data(), NumPES, HaEig, NumPES, 0.0, Hamil.data(), NumPES);
    // after multiplication there might be numerical error
    // since H is hermician, average the off-diagonal
    for (int i = 0; i < NumPES; i++)
    {
        for (int j = i + 1; j < NumPES; j++)
        {
            Hamil(i, j) = Hamil(j, i) = (Hamil(i, j) + Hamil(j, i)) / 2.0;
        }
    }
    return Hamil;
}

// second derivative of Hamiltonian under force basis;
// used to calculate the non-adiabatic coupling
static RealMatrix ForceHesse(const double x)
{
    // EigVec is diabatic "force" resized into 1D form before diag
    // and is the columnwise eigenvectors after diag
    // DiaH1D is diabatic Hessian of Hamiltonian resized into 1D form
    const RealMatrix&& EigVec = DiaToForceMatrix(x);
    // HeEig is the matrix, [Hesse] * [eigvec](column mat)
    double HeEig[NumPES * NumPES];
    // Hesse is the result matrix, [eigvec]^T * [Hesse] * [eigvec]
    RealMatrix Hesse(NumPES);
    // using d(ouble)sy(mmetic)/ge(neral)m(atrix)m(ultiplication) in CBLAS to calculate the above
    cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper, NumPES, NumPES, 1.0, DiaHesse(x).data(), NumPES, EigVec.data(), NumPES, 0.0, HeEig, NumPES);
    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, NumPES, NumPES, NumPES, 1.0, EigVec.data(), NumPES, HeEig, NumPES, 0.0, Hesse.data(), NumPES);
    // after multiplication there might be numerical error
    // since Hesse is hermician, average the off-diagonal
    for (int i = 0; i < NumPES; i++)
    {
        for (int j = i + 1; j < NumPES; j++)
        {
            Hesse(i, j) = Hesse(j, i) = (Hesse(i, j) + Hesse(j, i)) / 2.0;
        }
    }
    return Hesse;
}

// non-adiabatic coupling of force basis
RealMatrix ForceNAC(const double x)
{
    // NAC is the non-adiabatic coupling matrix
    // ForceH is the Hessian of Hamiltonian
    // ForceF is the diagonal "force"
    RealMatrix NAC(NumPES);
    const RealMatrix&& ForceH(ForceHesse(x)), ForceF(ForceForce(x));
    // dij=-Hesse_ij/(Fi-Fj) for off-diagonal elements
    // anti-hermiticity makes diagonal elements 0
    for (int i = 0; i < NumPES; i++)
    {
        for (int j = i + 1; j < NumPES; j++)
        {
            NAC(i, j) = -ForceH(i, j) / (ForceF(i, i) - ForceF(j, j));
            NAC(j, i) = -NAC(i, j);
        }
    }
    return NAC;
}


// the absorbing potential: V->V-iE
// Here is E only. E is diagonal
// E is zero in the interacting region
// E=hbar^2/2m*(2*pi/arl)^2*y(x)
// x=2*d*kmin*(r-r1)=c(r-r1)/arl
// d=c/4pi -> arl=2pi/kmin=h/pmin
// y(x)=sqrt(pow(cn(x/sqrt(2),1/sqrt(2)),-4)-1)
// ~4/(c-x)^2+4/(c+x)^2-8/c^2 from
// J. Chem. Phys., 2004, 120(5): 2247-2254,
// c=sqrt(2)*K(1/sqrt(2))
double AbsorbPotential(const double mass, const double xmin, const double xmax, const double AbsorbingRegionLength, const double r)
{
    static const double c = sqrt(2) * comp_ellint_1(1.0 / sqrt(2));
    // in the interacting region, no AP
    if (r > xmin && r < xmax)
    {
        return 0;
    }
    // otherwise, return E(x)_ii
    double x = c * (r < xmin ? r - xmin : r - xmax) / AbsorbingRegionLength;
    return pow(PlanckH / AbsorbingRegionLength, 2) * 2.0 / mass
        * (1.0 / pow(c - x, 2) + 1.0 / pow(c + x, 2) - 2.0 / pow(c, 2));
}

// the implementation of all the transformation matrices
void DiaToAdia(const int NGrids, double* GridCoordinate, Complex* Psi)
{
    Complex in[NumPES], out[NumPES];
    for (int i = 0; i < NGrids; i++)
    {
        for (int j = 0; j < NumPES; j++)
        {
            in[j] = Psi[j * NGrids + i];
        }
        cblas_zgemv(CblasRowMajor, CblasNoTrans, NumPES, NumPES, &Alpha, DiaToAdiaMatrix(GridCoordinate[i]).data(), NumPES,in, 1, &Beta, out, 1);
        for (int j = 0; j < NumPES; j++)
        {
            Psi[j * NGrids + i] = out[j];
        }
    }
}
void DiaToForce(const int NGrids, double* GridCoordinate, Complex* Psi)
{
    Complex in[NumPES], out[NumPES];
    for (int i = 0; i < NGrids; i++)
    {
        for (int j = 0; j < NumPES; j++)
        {
            in[j] = Psi[j * NGrids + i];
        }
        cblas_zgemv(CblasRowMajor, CblasNoTrans, NumPES, NumPES, &Alpha, DiaToForceMatrix(GridCoordinate[i]).data(), NumPES,in, 1, &Beta, out, 1);
        for (int j = 0; j < NumPES; j++)
        {
            Psi[j * NGrids + i] = out[j];
        }
    }
}
void AdiaToDia(const int NGrids, double* GridCoordinate, Complex* Psi)
{
    Complex in[NumPES], out[NumPES];
    for (int i = 0; i < NGrids; i++)
    {
        for (int j = 0; j < NumPES; j++)
        {
            in[j] = Psi[j * NGrids + i];
        }
        cblas_zgemv(CblasRowMajor, CblasConjTrans, NumPES, NumPES, &Alpha, DiaToAdiaMatrix(GridCoordinate[i]).data(), NumPES,in, 1, &Beta, out, 1);
        for (int j = 0; j < NumPES; j++)
        {
            Psi[j * NGrids + i] = out[j];
        }
    }
}
void AdiaToForce(const int NGrids, double* GridCoordinate, Complex* Psi)
{
    AdiaToDia(NGrids, GridCoordinate, Psi);
    DiaToForce(NGrids, GridCoordinate, Psi);
}
void ForceToDia(const int NGrids, double* GridCoordinate, Complex* Psi)
{
    Complex in[NumPES], out[NumPES];
    for (int i = 0; i < NGrids; i++)
    {
        for (int j = 0; j < NumPES; j++)
        {
            in[j] = Psi[j * NGrids + i];
        }
        cblas_zgemv(CblasRowMajor, CblasConjTrans, NumPES, NumPES, &Alpha, DiaToForceMatrix(GridCoordinate[i]).data(), NumPES,in, 1, &Beta, out, 1);
        for (int j = 0; j < NumPES; j++)
        {
            Psi[j * NGrids + i] = out[j];
        }
    }
}
void ForceToAdia(const int NGrids, double* GridCoordinate, Complex* Psi)
{
    ForceToDia(NGrids, GridCoordinate, Psi);
    DiaToAdia(NGrids, GridCoordinate, Psi);
}