// implementation of pes.h:
// diabatic PES and absorbing potential

#include <cmath>
#include "general.h"
#include "matrix.h"
#include "pes.h"

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
    Force(1, 1) = -Force(0, 0);
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

// transformation matrix from diabatic state to adiabatic state
// i.e. C^T*psi(dia)=psi(adia), which diagonalize PES only (instead of diagonal H)
ComplexMatrix DiaToAdia(const int NGrids, const double* GridCoordinate)
{
    ComplexMatrix TransformationMatrix(NGrids * NumPES);
    // EigVal stores the eigenvalues
    double EigVal[NumPES];
    
    for (int i = 0; i < NGrids; i++)
    {
        // EigVec stores the V before diagonalization and 
        // transformation matrix after diagonalization
        RealMatrix EigVec = DiaPotential(GridCoordinate[i]);
        // diagonal each grid
        if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', NumPES, EigVec.data(), NumPES, EigVal) > 0)
        {
            cerr << "FAILING DIAGONALIZE DIABATIC POTENTIAL AT " << GridCoordinate[i] << endl;
            exit(300);
        }
        // copy the transformation info to the whole matrix
        for (int j = 0; j < NumPES; j++)
        {
            for (int k = 0; k < NumPES; k++)
            {
                TransformationMatrix(j * NGrids + i, k * NGrids + i) = EigVec(j, k);
            }
        }
    }
    return TransformationMatrix;
}