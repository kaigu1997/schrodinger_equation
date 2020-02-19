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
    // PhaseOutput.sync_with_stdio(false);
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

    // read interaction region
    const double xmin = read_double(in);
    const double xmax = read_double(in);
    // read grid spacing, should be "~ 4 to 5 grids per de Broglie wavelength"
    // and then do the cut off, e.g. 0.2493 -> 0.2, 1.5364 -> 1
    // and the number of grids are thus determined
    const double dx = cutoff(min(read_double(in), PlanckH / p0max / 5.0));
    // grids in [xmin, xmax]
    const int InteractingGrid = static_cast<int>((xmax - xmin) / dx) + 1;
    // read whether have absorb potential or not
    const bool Absorbed = read_absorb(in);
    // absorbing region length from [2] and [3], determined by p0
    const double AbsorbingRegionLength = [&]
    {
        if (Absorbed == true)
        {
            return PlanckH / p0min;
        }
        else
        {
            return 0.0;
        }
    }();
    // total length is the interaction region + absorbing region
    const double TotalLength = xmax - xmin + 2.0 * AbsorbingRegionLength;
    // grids in [xmin-arl, xmin) or (xmax, xmax+arl]. no absorb -> 0 grids
    const int AbsorbingGrid = static_cast<int>(AbsorbingRegionLength / dx);
    // NGrids: number of grids in [xmin-arl, xmax+arl]
    const int NGrids = InteractingGrid + 2 * AbsorbingGrid;
    // dim: total number of elements (dimension) in Psi/H
    const int dim = NGrids * NumPES;
    // the coordinates of the grids, i.e. value of xi
    double* GridCoordinate = new double[NGrids];
    // calculate the grid coordinates, and print them
    for (int i = 0; i < NGrids; i++)
    {
        GridCoordinate[i] = xmin + dx * (i - AbsorbingGrid);
        Grids << GridCoordinate[i] << endl;
    }
    clog << "dx = " << dx << ", and there is overall " << NGrids << " grids from "
        << GridCoordinate[0] << " to " << GridCoordinate[NGrids - 1] << '.' << endl;
    Grids.close();

    // read evolving time and output time, in unit of a.u.
    const double TotalTime = read_double(in);
    const double PsiOutputTime = read_double(in);
    const double PhaseOutputTime = read_double(in);
    // read dt. criteria is from J. Comput. Phys., 1983, 52(1): 35-53.
    const double dt = [&]
    {
        if (Absorbed == false)
        {
            return PsiOutputTime;
        }
        else
        {
            return cutoff(min(read_double(in), 0.2 * 2.0 * mass * hbar / pow(p0max, 2)));
        }
    }();
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
    const ComplexMatrix Hamiltonian = Hamiltonian_construction(NGrids, GridCoordinate, dx, mass, Absorbed, xmin, xmax, AbsorbingRegionLength);
    // TransformationMatrix makes dia to adia
    const ComplexMatrix TransformationMatrix = DiaToAdia(NGrids, GridCoordinate);
    // assign for the diabatic and adiabatic wavefunction
    Complex* psi_t_dia = new Complex[dim];
    Complex* psi_t_adia = new Complex[dim];
    // construct the initial wavepacket: gaussian on the ground state PES
    // psi(x)=exp(-((x-x0)/2sigma_x)^2+i*p0*x/hbar)/sqrt(sqrt(2*pi)*sigma_x)
    wavefunction_initialization(NGrids, GridCoordinate, dx, x0, p0, SigmaX, psi_t_adia);
    // and transform to diabatic representation C^T*rho(dia)*C=rho(adia), so psi(dia)=C*psi(adia)
    cblas_zgemv(CblasRowMajor, CblasNoTrans, dim, dim, &Alpha, TransformationMatrix.data(), dim, psi_t_adia, 1, &Beta, psi_t_dia, 1);
    // population on each PES
    double Population[NumPES] = {1.0};


    // evolve
    // if H is hermitian, diagonal; otherwise, RK4
    if (Absorbed == false)
    {
        // diagonalize H
        ComplexMatrix EigVec = Hamiltonian;
        double* EigVal = new double[dim];
        if (LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', dim, reinterpret_cast<MKL_Complex16*>(EigVec.data()), dim, EigVal) > 0)
        {
            cerr << "FAILING DIAGONALIZE HAMILTONIAN WITHOUT ABSORBING POTENTIAL" << endl;
            exit(400);
        }
        // memory allocation:
        // psi(t)_adia=C2T*psi(t)_dia=C2T*C1*exp(-i*Hd*t)*C1T*psi(0)_dia
        // so we need: C2, C1(=EigVal), psi(t)_adia, psi(t)_dia, psi(t)_diag, psi(0)_diag
        // besides, we need to save the population on each PES
        // psi_t: diagonal representation. (a)diabatic have been allocated already
        Complex* psi_t_diag = new Complex[dim];
        // psi_0_diag: initial wavefunction of diagonal representation, =C1T*psi(0)_dia;
        Complex* psi_0 = new Complex[dim];
        cblas_zgemv(CblasRowMajor, CblasConjTrans, dim, dim, &Alpha, EigVec.data(), dim, psi_t_dia, 1, &Beta, psi_0, 1);
        const ComplexVector psi_0_diag(psi_0);
        // the population on each PES at last output moment
        double OldPopulation[NumPES] = {0};
        // before calculating population, check if wavepacket have passed through center(=0.0)
        bool PassedCenter = false;
        clog << "Finish diagonalization and memory allocation." << endl << show_time << endl;
        // evolution:
        for (int iStep = 0; iStep <= TotalStep; iStep += PsiOutputStep)
        {
            const double Time = iStep * dt;
            Steps << Time << endl;

            // calculate psi_t_diag = exp(-iH(diag)*t) * psi_0_diag,
            // each diagonal element be the eigenvalue
            for (int i = 0; i < dim; i++)
            {
                psi_t_diag[i] = exp(Complex(0.0, -EigVal[i] * Time)) * psi_0_diag[i];
            }
            // calculate psi_t_dia=C1*psi_t_diag
            cblas_zgemv(CblasRowMajor, CblasNoTrans, dim, dim, &Alpha, EigVec.data(), dim, psi_t_diag, 1, &Beta, psi_t_dia, 1);
            // calculate psi_t_adia=C2T*psi_t_dia
            cblas_zgemv(CblasRowMajor, CblasConjTrans, dim, dim, &Alpha, TransformationMatrix.data(), dim, psi_t_dia, 1, &Beta, psi_t_adia, 1);

            // print population on each grid
            for (int i = 0; i < dim; i++)
            {
                PsiOutput << ' ' << (psi_t_adia[i] * conj(psi_t_adia[i])).real();
            }
            PsiOutput << endl;

            /*/ check if calculating phase space distribution
            if (iStep % PhaseOutputStep == 0)
            {
                // print on the screen for monitoring
                clog << "t = " << Time << endl;
                // Wigner Transformation: P(x,p)=int{dy*exp(2ipy/hbar)<x-y|rho|x+y>}/(pi*hbar)
                // the interval of p is pi*hbar/dx*[-1,1), dp=2*pi*hbar/(xmax-xmin)
                output_phase_space_distribution(PhaseOutput, NGrids, GridCoordinate, dx, psi_t_adia);
            }// */

            // print population on each PES
            // and check if evolution should stop
            int NotChangedPES = 0;
            calculate_popultion(NGrids, dx, psi_t_adia, Population);
            // check if all PES are stable
            // first, check if the ground state wavepacket have been through origin
            if (PassedCenter == false)
            {
                // caculate the new center
                double xBar = 0;
                for (int i = 0; i < NGrids; i++)
                {
                    xBar += GridCoordinate[i] * (psi_t_adia[i] * conj(psi_t_adia[i])).real();
                }
                // then check if it passed the center
                if (xBar < 0)
                {
                    continue;
                }
                else
                {
                    PassedCenter = true;
                }
            }
            // after calculating new <x>, judge and check
            if (PassedCenter == true)
            {
                for (int i = 0; i < NumPES; i++)
                {
                    if (abs(Population[i] - OldPopulation[i]) < ChangeLim)
                    {
                        NotChangedPES++;
                    }
                }
                if (NotChangedPES == NumPES)
                {
                    cerr << "POPULATION ON EACH PES IS STABLE. STOP EVOLVING AT " << Time << endl;
                    break;
                }
            }
            // after checking, copy the population
            copy(Population, Population + NumPES, OldPopulation);
        }
        // after evolution, frees the resources
        delete[] psi_t_diag;
        delete[] EigVal;
        // and output
        clog << "Finish evolution." << endl << show_time << endl << endl;
        /*/ model 1 and 3
        cout << p0; // */
        // model 2
        cout << log(p0 * p0 / 2.0 / mass);
        for (int i = 0; i < NumPES; i++)
        {

        } // */
        cout << endl;
    }
    else
    {
        // RK4
        // prepare memory for RK4
        Complex* kIncrement = new Complex[dim];
        Complex* kIncrementMatVec = new Complex[dim];
        Complex* psi_new = new Complex[dim];
        memcpy(psi_new, psi_t_dia, dim * sizeof(Complex));
        const double RK4Parameter[] = {1.0, 2.0, 2.0, 1.0};
        clog << "Finish initialization. Begin evolving." << endl << show_time << endl;
        // evolve
        for (int iStep = 0; iStep <= TotalStep; iStep++)
        {
            // output the time
            const double Time = iStep * dt;
            // check if output the wavefunction
            if (iStep % PsiOutputStep == 0)
            {
                Steps << Time << endl;
                // transform to adiabatic representation
                cblas_zgemv(CblasRowMajor, CblasConjTrans, dim, dim, &Alpha, TransformationMatrix.data(), dim, psi_t_dia, 1, &Beta, psi_t_adia, 1);

                // print population on each grid
                for (int i = 0; i < dim; i++)
                {
                    PsiOutput << ' ' << (psi_t_adia[i] * conj(psi_t_adia[i])).real();
                }
                PsiOutput << endl;

                /*/ check if calculating phase space distribution
                if (iStep % PhaseOutputStep == 0)
                {
                    // print on the screen for monitoring
                    clog << "t = " << Time << endl;
                    // Wigner Transformation: P(x,p)=int{dy*exp(2ipy/hbar)<x-y|rho|x+y>}/(pi*hbar)
                    // the interval of p is pi*hbar/dx*[-1,1), dp=2*pi*hbar/(xmax-xmin)
                    output_phase_space_distribution(PhaseOutput, NGrids, GridCoordinate, dx, psi_t_adia);
                }// */

                // calculate the adiabatic population, and check if finish the program (absorb enough)
                calculate_popultion(NGrids, dx, psi_t_adia, Population);
                if (accumulate(Population, Population + NumPES, 0.0) < PplLim)
                {
                    break;
                }
            }

            // evolve, in diabatic representation, using RK4
            // k1=f(y,t), k2=f(y+dt/2*k1, t+dt/2), k3=f(y+dt/2*k2,t+dt/2)
            // k4=f(y+dt*k3, t+dt), y(t+dt)=y(t)+dt/6*(k1+2*k2+2*k3+k4)
            // here f(y,t)=hat(H)/i/hbar*psi
            memset(kIncrement, 0, dim * sizeof(Complex));
            for (int i = 0; i < 4; i++)
            {
                const Complex RK4kBeta = dt / RK4Parameter[i] * RK4kAlpha;
                const Complex RK4PsiAlpha(dt / 6.0 * RK4Parameter[i]);
                // k(i)=(psi(t)+dt/(1,2)*k(i-1))/i/hbar
                cblas_zaxpby(dim, &RK4kAlpha, psi_t_dia, 1, &RK4kBeta, kIncrement, 1);
                // k'(i)=H*k(i)
                cblas_zgemv(CblasRowMajor, CblasNoTrans, dim, dim, &Alpha, Hamiltonian.data(), dim, kIncrement, 1, &Beta, kIncrementMatVec, 1);
                // change the k' and k
                swap(kIncrement, kIncrementMatVec);
                // add to the wavefunction
                cblas_zaxpy(dim, &RK4PsiAlpha, kIncrement, 1, psi_new, 1);
            }
            memcpy(psi_t_dia, psi_new, dim * sizeof(Complex));
        }
        // after evolution, frees the resources
        delete[] psi_new;
        delete[] kIncrementMatVec;
        delete[] kIncrement;
        // and output
        clog << "Finish evolution." << endl << show_time << endl << endl;
        /*/ model 1 and 3
        cout << p0; // */
        // model 2
        cout << log(p0 * p0 / 2.0 / mass);
        for (int i = 0; i < NumPES; i++)
        {

        } // */
        cout << endl;
    }

    // end. free the memory, close the files.
    delete[] psi_t_adia;
    delete[] psi_t_dia;
    delete[] GridCoordinate;
    Steps.close();
    // PhaseOutput.close();
    PsiOutput.close();
	return 0;
}
