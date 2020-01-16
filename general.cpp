// implementation of general.h:
// diabatic PES and Hamiltonians

#include <algorithm>
#include <cmath>
#include <iostream>
#include <mkl.h>
#include "general.h"
#include "matrix.h"

//*Model 1
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

/*Model 2
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