// The purpose of this program is to give
// an exact solution of quantum mechanic problem
// using Discrete Variable Representation (DVR)
// in [1]J. Chem. Phys., 1992, 96(3): 1982-1991,
// with Absorbing Boundary Condition in
// [2]J. Chem. Phys., 2002, 117(21): 9552-9559
// and [3]J. Chem. Phys., 2004, 120(5): 2247-2254.
// This program could be used to solve
// exact solution under diabatic basis ONLY.
// It requires C++17 or newer C++ standards when compiling
// and needs connection to Intel(R) Math Kernel Library
// (MKL) by whatever methods: icpc/msvc/gcc -I.
// Error code criteria: 1XX for matrix, 
// 2XX for general, 3XX for pes, and 4XX for main.

#include <chrono>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mkl.h>
#include <numeric>
#include <utility>
#include "general.h"
#include "pes.h"
#include "matrix.h"
using namespace std;

int main(void)
{
    // initialize: read input and calculate cooresponding constants
    // including the number of grids, their coordinates, etc
    cout.sync_with_stdio(false);
    clog.sync_with_stdio(false);
    cerr.sync_with_stdio(false);
    // psi: the wavefunction, phase: the phase space distribution
    // In psi, each line is the wavefunction at a moment. In each line,
    // the order is t, psi(t)[0].real, psi(t)[0].imag, psi(t)[1].real, ...
    // In Phase space, each line is the PS-distribution at a moment:
    // t, P(x0,p0,t), P(x1,p0,t), ... P(x0,p1,t)...
    ofstream PsiOutput("psi.txt");
    PsiOutput.sync_with_stdio(false);
    // ofstream PhaseOutput("phase.txt");
    // Grids contains each grid coordinate, one in a line
    // Steps contains when is each step, also one in a line
    ofstream Grids("x.txt"), Steps("t.txt");
    Grids.sync_with_stdio(false);
    Steps.sync_with_stdio(false);
    // in: the input file
    ifstream in("input");
    in.sync_with_stdio(false);
    // read mass: the mass of the bath
    const double mass = read_double(in);
    // read initial wavepacket info
    // the center/width of the wavepacket
    // calculate SigmaX by SigmaP using minimum uncertainty rule
    const double x0 = read_double(in);
    const double p0 = read_double(in);
    const double SigmaP = read_double(in);
    const double SigmaX = hbar / 2.0 / SigmaP;
    clog << "The particle weighes " << mass << " a.u.," << endl
        << "starting from " << x0 << " with initial momentum " << p0 << '.' << endl
        << "Initial width of x and p are " << SigmaX << " and " << SigmaP << ", respectively." << endl;
    // 99.7% initial momentum in this region: p0+-3SigmaP
    // calculate the region of momentum by p0 and SigmaP
    const double p0min = p0 - 3.0 * SigmaP;
    const double p0max = p0 + 3.0 * SigmaP;

    // read the size of the box
    const double xmin = read_double(in);
    const double xmax = read_double(in);
    // total length is the interaction region + absorbing region
    const double TotalLength = xmax - xmin;
    // read the interaction region
    const double InteractionRegionLeft = read_double(in);
    const double InteractionRegionRight = read_double(in);
    const double InteractionRegionLength = InteractionRegionRight - InteractionRegionLeft;
    // read grid spacing, should be "~ 4 to 5 grids per de Broglie wavelength"
    // and then do the cut off, e.g. 0.2493 -> 0.2, 1.5364 -> 1
    // and the number of grids are thus determined
    const double dx = cutoff(min(read_double(in), PlanckH / p0max / 5.0));
    // NGrids: number of grids in [xmin, xmax]
    const int NGrids = static_cast<int>((xmax - xmin) / dx) + 1;
    // get the index of left/right boundary of the interaction region
    const int InteractionLeftIndex = static_cast<int>((InteractionRegionLeft - xmin) / dx);
    const int InteractionRightIndex = static_cast<int>((InteractionRegionRight - xmin) / dx);
    // dim: total number of elements (dimension) in Psi/H
    const int dim = NGrids * NumPES;
    // the coordinates of the grids, i.e. value of xi
    double* GridCoordinate = new double[NGrids];
    // calculate the grid coordinates, and print them
    for (int i = 0; i < NGrids; i++)
    {
        GridCoordinate[i] = xmin + dx * i;
        Grids << GridCoordinate[i] << endl;
    }
    clog << "dx = " << dx << ", and there is overall " << NGrids << " grids" << endl
        << "in the region [" << xmin << ',' << xmax << "]," << endl
        << "and the interaction is limited in [" << InteractionRegionLeft << ',' << InteractionRegionRight << "]." << endl;
    Grids.close();

    // read evolving time and output time, in unit of a.u.
    // read dt. criteria is from J. Comput. Phys., 1983, 52(1): 35-53.
    const double dt = cutoff(min(read_double(in), 0.2 * 2.0 * mass * hbar / pow(p0max, 2)));
    // total evolving time and output time, in unit of a.u.
    const double TotalTime = read_double(in);
    const double PsiOutputTime = read_double(in);
    const double PhaseOutputTime = read_double(in); 
    // finish reading
    in.close();
    // calculate corresponding dt of the above (how many dt they have)
    const int TotalStep = static_cast<int>(TotalTime / dt);
    const int PsiOutputStep = static_cast<int>(PsiOutputTime / dt);
    // when output phase space, should also output Psi for less calculation
    const int PhaseOutputStep = static_cast<int>(PhaseOutputTime / PsiOutputTime) * PsiOutputStep;
    clog << "dt = " << dt << ", and there is overall " << TotalStep << " time steps." << endl;

    // construct the Hamiltonian. n'=nn, m'=mm
    // diabatic Hamiltonian used for propagator
    // dc/dt=-iHc/hbar => c(t)=e^(-iHt)c(0)
    // c is the coefficient, c[m*NGrids+n] is
    // the nth grid on the mth surface
    const ComplexMatrix Hamiltonian = Hamiltonian_construction(NGrids, GridCoordinate, dx, mass);
    // transformation matrix: M*psi(dia)=psi(adia)
    const ComplexMatrix TransformationMatrix = DiaToAdia(NGrids, GridCoordinate);
    // assign for the diabatic/adiabatic wavefunction
    Complex* psi_dia = new Complex[dim];
    Complex* psi_adia = new Complex[dim];
    // ... and for RK4
    Complex* psi_new = new Complex[dim];
    // construct the initial wavepacket: gaussian on the ground state PES
    // psi(x)=exp(-((x-x0)/2sigma_x)^2+i*p0*x/hbar)/sqrt(sqrt(2*pi)*sigma_x)
    diabatic_wavefunction_initialization(NGrids, GridCoordinate, dx, x0, p0, SigmaX, psi_adia);
    cblas_zgemv(CblasRowMajor, CblasNoTrans, dim, dim, &Alpha, TransformationMatrix.data(), dim, psi_adia, 1, &Beta, psi_dia, 1);
    // assign for manual absorption, data: psi[0].left, psi[1].left,
    // ..., psi[n].left, psi[0].right, psi[1].right, ...
    double AbsorbedPopulation[2 * NumPES] = {0};
    // the population still in interaction region
    double InteractionPopulation[NumPES] = {0};
    // preparation for RK4
    Complex* kIncrement = new Complex[dim];
    Complex* kIncrementMatVec = new Complex[dim];
    const double RK4Parameter[] = {1.0, 2.0, 2.0, 1.0};
    clog << "Finish initialization. Begin evolving." << endl << show_time;

    // evolve
    for (int iStep = 0; iStep <= TotalStep; iStep++)
    {
        // output the time
        const double Time = iStep * dt;
        // cutoff: manually absorb in adiabatic representation
        cblas_zgemv(CblasRowMajor, CblasConjTrans, dim, dim, &Alpha, TransformationMatrix.data(), dim, psi_dia, 1, &Beta, psi_adia, 1);
        absorb(NGrids, InteractionLeftIndex, InteractionRightIndex, dx, psi_adia, AbsorbedPopulation);
        cblas_zgemv(CblasRowMajor, CblasNoTrans, dim, dim, &Alpha, TransformationMatrix.data(), dim, psi_adia, 1, &Beta, psi_dia, 1);
        
        // check if output the wavefunction
        if (iStep % PsiOutputStep == 0)
        {
            Steps << Time << endl;
            // print population on each grid
            for (int i = 0; i < dim; i++)
            {
                PsiOutput << ' ' << (psi_adia[i] * conj(psi_adia[i])).real();
            }
            PsiOutput << endl;
            /*/ check if calculating phase space distribution
            // problematic, the calculation is wrong
            if (iStep % PhaseOutputStep == 0)
            {
                // print on the screen for monitoring
                clog << "t = " << Time << endl;
                PhaseOutput << Time;
                // Wigner Transformation: P(x,p)=int{dy*exp(2ipy/hbar)<x-y|rho|x+y>}/(pi*hbar)
                // the interval of p is pi*hbar/dx*[-1,1), dp=2*pi*hbar/(xmax-xmin)
                // loop over p first
                for (int i = 0; i < NGrids; i++)
                {
                    const double p = (i - NGrids / 2) * 2 * pi * hbar / TotalLength;
                    // loop over x
                    for (int j = 0; j < NGrids; j++)
                    {
                        // do the numerical integral and output
                        Complex integral;
                        for (int k = max(-j, j + 1 - NGrids); k <= min(j, NGrids - 1 - j); k++)
                        {
                            integral += exp(2.0i * p * GridCoordinate[k] / hbar) * conj(Psi0[j + k]) * Psi0[j - k];
                        }
                        PhaseOutput << ' ' << integral.real() / pi / hbar;
                    }
                }
                PhaseOutput << endl;
            }// */
        }        
        
        // calculate the adiabatic population, and check if finish the program (absorb enough)
        calculate_popultion(NGrids, dx, psi_adia, InteractionPopulation);
        if (accumulate(InteractionPopulation, InteractionPopulation + NumPES, 0.0) < PplLim)
        {
            break;
        }

        // evolve, in diabatic representation, using RK4
        memset(kIncrement, 0, dim * sizeof(Complex));
        memcpy(psi_new, psi_dia, dim * sizeof(Complex));
        for (int i = 0; i < 4; i++)
        {
            const Complex RK4kBeta = dt / RK4Parameter[i] * RK4kAlpha;
            const Complex RK4PsiAlpha(dt / 6.0 * RK4Parameter[i]);
            // k(i)=psi(t)+dt/(1,2)*k(i-1)
            cblas_zaxpby(dim, &RK4kAlpha, psi_dia, 1, &RK4kBeta, kIncrement, 1);
            // k(i)=H*k(i)
            cblas_zhemv(CblasRowMajor, CblasUpper, dim, &Alpha, Hamiltonian.data(), dim, kIncrement, 1, &Beta, kIncrementMatVec, 1);
            // change the two
            swap(kIncrement, kIncrementMatVec);
            // add to the wavefunction
            cblas_zaxpy(dim, &RK4PsiAlpha, kIncrement, 1, psi_new, 1);
        }
        swap(psi_new, psi_dia);
    }
    // print the final info
    cout << log(p0 * p0 / 2.0 / mass);
    for (int i = 0; i < 2 * NumPES; i++)
    {
        cout << ' ' << AbsorbedPopulation[i];
    }
    cout << endl;
    // after evolution, print time and frees the resources
    clog << "Finish evolution." << endl << show_time << endl << endl;    
    

    // end. free the memory, close the files.
    delete[] kIncrement;
    delete[] kIncrementMatVec;
    delete[] psi_adia;
    delete[] psi_dia;
    delete[] psi_new;
    delete[] GridCoordinate;
    Steps.close();
    // PhaseOutput.close();
    PsiOutput.close();
	return 0;
}
