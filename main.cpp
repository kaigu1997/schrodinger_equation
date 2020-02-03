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
#include "general.h"
#include "pes.h"
#include "matrix.h"
using namespace std;

int main(void)
{
    // initialize: read input and calculate cooresponding constants
    // including the number of grids, their coordinates, etc
    // psi: the wavefunction, phase: the phase space distribution
    // In psi, each line is the wavefunction at a moment. In each line,
    // the order is t, psi(t)[0].real, psi(t)[0].imag, psi(t)[1].real, ...
    // In Phase space, each line is the PS-distribution at a moment:
    // t, P(x0,p0,t), P(x1,p0,t), ... P(x0,p1,t)...
    ofstream PsiOutput("psi.txt");
    // ofstream PhaseOutput("phase.txt");
    // Grids contains each grid coordinate, one in a line
    // Steps contains when is each step, also one in a line
    ofstream Grids("x.txt"), Steps("t.txt");
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
    // 99.7% initial momentum in this region: p0+-3SigmaP
    // calculate the region of momentum by p0 and SigmaP
    const double p0min = p0 - 3.0 * SigmaP;
    const double p0max = p0 + 3.0 * SigmaP;
    // read interaction region
    const double xmin = read_double(in);
    const double xmax = read_double(in);
    clog << "The particle weighes " << mass << " a.u.," << endl
        << "starting from " << x0 << " with initial momentum " << p0 << '.' << endl
        << "Initial width of x and p are " << SigmaX << " and " << SigmaP << ", respectively." << endl;
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
    clog << "dx = " << dx << ", and there is overall " << NGrids << " grids." << endl;
    Grids.close();

    // read evolving time and output time, in unit of a.u.
    const double TotalTime = read_double(in);
    const double PsiOutputTime = read_double(in);
    const double PhaseOutputTime = read_double(in);
    const double dt = PsiOutputTime;
    // finish reading
    in.close();
    // calculate corresponding dt of the above (how many dt they have)
    // when output phase space, should also output Psi for less calculation
    const int TotalStep = static_cast<int>(TotalTime / dt);
    const int PsiOutputStep = static_cast<int>(PsiOutputTime / dt);
    const int PhaseOutputStep = static_cast<int>(PhaseOutputTime / PsiOutputTime) * PsiOutputStep;
    clog << "dt = " << dt << ", and there is overall " << TotalStep << " time steps." << endl;

    // construct the initial wavepacket: gaussian on the ground state PES
    // psi(x)=exp(-((x-x0)/2sigma_x)^2+i*p0*x/hbar)/sqrt(sqrt(2*pi)*sigma_x)
    const ComplexVector Psi0 = wavefunction_initialization(NGrids, GridCoordinate, dx, x0, p0, SigmaX);
    // construct the Hamiltonian. n'=nn, m'=mm
    // diabatic Hamiltonian used for propagator
    // dc/dt=-iHc/hbar => c(t)=e^(-iHt)c(0)
    // c is the coefficient, c[m*NGrids+n] is
    // the nth grid on the mth surface
    const ComplexMatrix Hamiltonian = Hamiltonian_construction(NGrids, GridCoordinate, dx, mass, Absorbed, xmin, xmax, AbsorbingRegionLength);
    clog << "Finish initialization. Begin evolving." << endl << show_time << endl;


    // evolve; if H is hermitian, diagonal; otherwise, Schur decomposition
    // and see Parlett B N. Linear Algebra Appl., 1976, 14(2): 117-121.
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
        // so we need: psi(t)_adia, psi(t)_dia, C2, C1(=EigVal), exp(-i*Hd*t), exp(-i*Hd*t)*C1
        // besides, we need to save the population on each PES
        // psi_t: diabatic/adiabatic representation
        Complex* psi_t_dia = new Complex[dim];
        Complex* psi_t_adia = new Complex[dim];
        // EigPropa is an intermediate, C1*exp(-i*Hd*t)
        // original representation exp(-iHdt)=PropaEig*C1T
        Complex* PropaEig = new Complex[dim * dim];
        // Propagator is exp(-iHdt)
        ComplexMatrix Propagator(dim);
        // TransformationMatrix makes dia to adia
        const ComplexMatrix TransformationMatrix = DiaToAdia(NGrids, GridCoordinate);
        // population on each PES, and the population on each PES at last output moment
        double Population[NumPES] = {1.0}, OldPopulation[NumPES] = {0};
        // before calculating population, check if wavepacket have passed through center(=0.0)
        bool PassedCenter = false;
        clog << "Finish diagonalization and memory allocation." << endl << show_time << endl;
        // evolution:
        for (int iStep = 0; iStep <= TotalStep; iStep += PsiOutputStep)
        {
            const double Time = iStep * dt;
            Steps << Time << endl;

            // calculate exp(-iH(diag)dt)
            // set all the matrix elements to be zero
            memset(Propagator.data(), 0, dim * dim * sizeof(Complex));
            // each diagonal element be exp(-iH*dt)
            for (int i = 0; i < dim; i++)
            {
                Propagator(i, i) = exp(Complex(0.0, -EigVal[i] * Time));
            }
            // then transform back to the diabatic basis, exp(-iHt)=C1*exp(-i*Hd*t)*C1T
            cblas_zsymm(CblasRowMajor, CblasRight, CblasUpper, dim, dim, &Alpha, Propagator.data(), dim, EigVec.data(), dim, &Beta, PropaEig, dim);
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, dim, dim, dim, &Alpha, PropaEig, dim, EigVec.data(), dim, &Beta, Propagator.data(), dim);
            // calculate psi_t_dia=psi(t)_dia=exp(-iHt)_dia*psi(0)_dia
            cblas_zgemv(CblasRowMajor, CblasNoTrans, dim, dim, &Alpha, Propagator.data(), dim, Psi0.get(), 1, &Beta, psi_t_dia, 1);
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
                    // print the final info
                    cout << log(p0 * p0 / 2.0 / mass);
                    for (int i = 0; i < NumPES; i++)
                    {
                        cout << ' ' << Population[i];
                    }
                    cout << endl;
                    break;
                }
            }
            // after checking, copy the population
            copy(Population, Population + NumPES, OldPopulation);
        }
        // after evolution, print time and frees the resources
        clog << "Finish evolution." << endl << show_time << endl;
        delete[] PropaEig;
        delete[] psi_t_adia;
        delete[] psi_t_dia;
        delete[] EigVal;
    }
    else
    {
        // Schur decomposition of H
        // In LAPACK, there is only generalized Schur decomposition,
        // which upper-trianglize A and B together
        // Here we set B=I, vsl and vsr are the left/right eigenvectors
        ComplexMatrix SchurForm = Hamiltonian;
        ComplexMatrix Identity(dim);
        for (int i = 0; i < dim; i++)
        {
            Identity(i, i) = 1.0;
        }
        Complex* vsl = new Complex[dim * dim];
        Complex* vsr = new Complex[dim * dim];
        Complex* EigenA = new Complex[dim];
        Complex* EigenB = new Complex[dim];
        int sdim;
        if (LAPACKE_zgges(LAPACK_ROW_MAJOR, 'V', 'V', 'N', nullptr, dim, reinterpret_cast<MKL_Complex16*>(SchurForm.data()), dim, reinterpret_cast<MKL_Complex16*>(Identity.data()), dim, &sdim, reinterpret_cast<MKL_Complex16*>(EigenA), reinterpret_cast<MKL_Complex16*>(EigenB), reinterpret_cast<MKL_Complex16*>(vsl), dim, reinterpret_cast<MKL_Complex16*>(vsr), dim) > 0)
        {
            cerr << "FAILING CALCULATING SCHUR DECOMPOSITION OF HAMILTONIAN WITH ABSORBING POTENTIAL" << endl;
            exit(401);
        }
        // After having the Schur form and the transformation matrix (vsl, vsr), the eigenvalues are not needed
        delete[] EigenA;
        delete[] EigenB;
        // memory allocation:
        // psi(t)_adia=C2T*psi(t)_dia=C2T*C1*exp(-i*Hd*t)*C1T*psi(0)_dia
        // so we need: psi(t)_adia, psi(t)_dia, C2, C1(=EigVal), exp(-i*Hd*t), exp(-i*Hd*t)*C1
        // besides, we need to save the population on each PES
        // psi_t: diabatic/adiabatic representation
        Complex* psi_t_dia = new Complex[dim];
        Complex* psi_t_adia = new Complex[dim];
        // EigPropa is an intermediate, C1*exp(-i*Hd*t)
        // original representation exp(-iHdt)=PropaEig*C1T
        Complex* PropaEig = new Complex[dim * dim];
        // Propagator is exp(-iHdt)
        ComplexMatrix Propagator(dim);
        // TransformationMatrix makes dia to adia
        const ComplexMatrix TransformationMatrix = DiaToAdia(NGrids, GridCoordinate);
        // population on each PES
        double Population[NumPES];
        // evolution:
        for (int iStep = 0; iStep <= TotalStep; iStep += PsiOutputStep)
        {
            const double Time = iStep * dt;
            Steps << Time << endl;

            // calculate exp(-iH(diag)dt)
            // set all the matrix elements to be zero
            memset(Propagator.data(), 0, dim * dim * sizeof(Complex));
            // each diagonal element be exp(-iH*dt)
            for (int i = 0; i < dim; i++)
            {
                Propagator(i, i) = exp(-1.0i * SchurForm(i, i) * Time);
            }
            // for the off-diagonal elements, denot P=exp(-iHt), then for r<s
            // H_{rr}P_{rs}-P_{rs}H_{ss}=\sum_{k=0}^{s-r-1}{P_{r,r+k}H_{r+k,s}-H_{r,s-k}P_{s-k,s}}
            // goes from diagonal to (0,1)->(n-1,n), then (0,2)->(n-2,n), ... so on
            // for the for-loops below, i=s-r
            for (int i = 1; i < dim; i++)
            {
                for (int r = 0, s = i; s < dim; r++, s++)
                {
                    Complex sum = 0;
                    for (int k = 0; k < i; k++)
                    {
                        sum += Propagator(r, r+k) * SchurForm(r+k, s) - SchurForm(r, s-k) * Propagator(s-k, s);
                    }
                    Propagator(r, s) = sum / (SchurForm(r, r) - SchurForm(s, s));
                }
            }
            // then transform back to the diabatic basis, exp(-iHt)=C1*exp(-i*Hut*t)*C1T
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, &Alpha, vsl, dim, Propagator.data(), dim, &Beta, PropaEig, dim);
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, dim, dim, dim, &Alpha, PropaEig, dim, vsr, dim, &Beta, Propagator.data(), dim);
            // calculate psi_t_dia=psi(t)_dia=exp(-iHt)_dia*psi(0)_dia
            cblas_zgemv(CblasRowMajor, CblasNoTrans, dim, dim, &Alpha, Propagator.data(), dim, Psi0.get(), 1, &Beta, psi_t_dia, 1);
            // calculate psi_t_adia=C2T*psi_t_dia
            cblas_zgemv(CblasRowMajor, CblasConjTrans, dim, dim, &Alpha, TransformationMatrix.data(), dim, psi_t_dia, 1, &Beta, psi_t_adia, 1);

            // print population on each grid
            for (int i = 0; i < dim; i++)
            {
                PsiOutput << ' ' << (psi_t_adia[i] * conj(psi_t_adia[i])).real();
            }
            PsiOutput << endl;

            // print population on each PES
            calculate_popultion(NGrids, dx, psi_t_adia, Population);
            cout << Time;
            for (int i = 0; i < NumPES; i++)
            {
                cout << ' ' << Population[i];
            }
            cout << endl;
            
            /*/ check if calculating phase space distribution
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
            // if the total population has been absorbed too much, stop evolution
            if (accumulate(Population, Population + NumPES, 0.0) < 1.0 - AbsorbLim)
            {
                cerr << "TOO MUCH POPULATION HAVE BEEN ABSORBED. STOP EVOLVING." << endl;
                break;
            }
        }
        // after evolution, print time and frees the resources
        clog << "Finish evolution." << endl << show_time << endl << endl << endl;
        delete[] PropaEig;
        delete[] psi_t_adia;
        delete[] psi_t_dia;
        delete[] vsr;
        delete[] vsl;
    }
    

    // end. free the memory, close the files.
    delete[] GridCoordinate;
    Steps.close();
    // PhaseOutput.close();
    PsiOutput.close();
	return 0;
}
