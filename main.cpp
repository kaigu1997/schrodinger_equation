// The purpose of this program is to give
// an exact solution of quantum mechanic problem
// using Discrete Variable Representation (DVR)
// in [1]J. Chem. Phys., 1992, 96(3): 1982-1991,
// with Absorbing Boundary Condition in
// [2]J. Chem. Phys., 2002, 117(21): 9552-9559
// and [3]J. Chem. Phys., 2004, 120(5): 2247-2254.
// This program could be used to solve
// exact solution under diabatic/adiabatic/force basis.
// It requires C++17 or newer C++ standards when compiling
// and needs connection to Intel(R) Math Kernel Library
// (MKL) by whatever methods: icpc/msvc/gcc -l.
// Error code criteria: 100-199 for matrix, 
// 200-299 for general, and 300-399 for for main.

#include <algorithm>
#include <chrono>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mkl.h>
#include "general.h"
#include "matrix.h"
using namespace std;

// read a double: mass, x0, etc
static inline double read_double(istream& is)
{
    static string buffer;
    static double temp;
    getline(is, buffer);
    is >> temp;
    getline(is, buffer);
    return temp;
}

// check if absorbing potential is used or not
static inline bool read_absorb(istream& is)
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

// check if absorbing potential is used 
static inline Representation read_represent(istream& is)
{
    string buffer;
    string Represent;
    Representation HamiltonianType;
    getline(is, buffer);
    is >> Represent;
    getline(is, buffer);
    if (strcmp(Represent.c_str(), "diabatic") == 0)
    {
        HamiltonianType = Diabatic;
    }
    else if (strcmp(Represent.c_str(), "adiabatic") == 0)
    {
        HamiltonianType = Adiabatic;
    }
    else if (strcmp(Represent.c_str(), "force") == 0)
    {
        HamiltonianType = Force;
    }
    else
    {
        cerr << "UNKNOWN REPRESENTATION TO DO DYNAMICS" << endl;
        exit(302);
    }
    return HamiltonianType;
}

// do the cutoff, e.g. 0.2493 -> 0.2, 1.5364 -> 1
static inline double cutoff(const double val)
{
    double pownum = pow(10, static_cast<int>(floor(log10(val))));
    return static_cast<int>(val / pownum) * pownum;
}

// return the normalization factor
static inline void normalization(Complex* Psi, const int size)
{
    Complex PsiSquare;
    cblas_zdotc_sub(size, Psi, 1, Psi, 1, &PsiSquare);
    double NormFactor = sqrt(PsiSquare.real());
    for (int i = 0; i < size; i++)
    {
        Psi[i] /= NormFactor;
    }
}

// to print current time
ostream& show_time(ostream& os)
{
    auto time = chrono::system_clock::to_time_t(chrono::system_clock::now());
    os << ctime(&time);
    return os;
}

int main(void)
{
    // initialize: read input and calculate cooresponding constants
    // including the number of grids, their coordinates, etc
    // psi: the wavefunction, phase: the phase space distribution
    // In psi, each line is the wavefunction at a moment. In each line,
    // the order is t, psi(t)[0].real, psi(t)[0].imag, psi(t)[1].real, ...
    // In Phase space, each line is the PS-distribution at a moment:
    // t, P(x0,p0,t), P(x1,p0,t), ... P(x0,p1,t)...
    ofstream PsiOutput("psi.txt"), PhaseOutput("phase.txt");
    // in: the input file
    ifstream in("input");
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

    // read whether have absorb potential or not
    const bool Absorbed = read_absorb(in);
    // absorbing region length from [2] and [3], determined by p0
    const double AbsorbingRegionLength = PlanckH / p0min;
    // read grid spacing, should be "~ 4 to 5 grids per de Broglie wavelength"
    // and then do the cut off, e.g. 0.2493 -> 0.2, 1.5364 -> 1
    // and the number of grids are thus determined
    const double dx = cutoff(min(read_double(in), PlanckH / p0max / 5.0));
    // grids in [xmin, xmax]
    const int InteractingGrid = static_cast<int>((xmax - xmin) / dx) + 1;
    // grids in [xmin-arl, xmin) or (xmax, xmax+arl]. no absorb -> 0 grids
    const int AbsorbingGrid = (Absorbed == true ? static_cast<int>(AbsorbingRegionLength / dx) : 0);
    // NGrids: number of grids in [xmin-arl, xmax+arl]
    // dim: total number of elements (dimension) in Psi/H
    const int NGrids = InteractingGrid + 2 * AbsorbingGrid;
    const int dim = NGrids * NumPES;
    // the coordinates of the grids, i.e. value of xi
    double* GridCoordinate = new double[NGrids];
    // calculate the grid coordinates
    for (int i = 0; i < NGrids; i++)
    {
        GridCoordinate[i] = xmin + dx * (i - AbsorbingGrid);
    }
    clog << "dx = " << dx << ", and there is overall " << NGrids << " Grids." << endl;
    // construct the initial wavepacket: gaussian on the ground state PES
    // psi(x)=exp(-((x-x0)/2sigma_x)^2+i*p0*x/hbar)/sqrt(sqrt(2*pi)*sigma_x)
    // as on the grids, the normalization needs redoing
    Complex* Psi = new Complex[dim];
    memset(Psi, 0, dim * sizeof(Complex));
    for (int i = 0; i < NGrids; i++)
    {
        Psi[i] = exp(p0 * GridCoordinate[i] / hbar * 1.0i) * exp(-pow((GridCoordinate[i] - x0) / 2 / SigmaX, 2));
    }
    normalization(Psi, dim);

    // read dt. criteria is from J. Comput. Phys., 1983, 52(1): 35-53.
    const double dt = cutoff(min(read_double(in), 0.2 * 2.0 * mass * hbar / pow(p0max, 2)));
    clog << "dt = " << dt << endl;
    // total evolving time and output time, in unit of a.u.
    const double TotalTime = read_double(in);
    const double PsiOutputTime = read_double(in);
    const double PhaseOutputTime = read_double(in);
    // calculate corresponding dt of the above (how many dt they have)
    // when output phase space, should also output Psi for less calculation
    const int TotalStep = static_cast<int>(TotalTime / dt) + 1;
    const int PsiOutputStep = static_cast<int>(PsiOutputTime / dt) + 1;
    const int PhaseOutputStep = static_cast<int>(PhaseOutputTime / PsiOutputTime) * (PsiOutputStep - 1) + 1;

    // read representation where doing dynamics
    const Representation HamiltonianType = read_represent(in);
    // finish reading
    in.close();
    // construct the Hamiltonian. n'=nn, m'=mm
    // diabatic/adiabatic/force-basis Hamiltonian
    // used for propagator: dc/dt=-iHc/hbar
    // c is the coefficient, c[m*NGrids+n] is
    // the nth grid on the mth surface
    ComplexMatrix Hamiltonian(dim);
    // 1. V_{mm'}(R_n), n=n'
    for (int n = 0; n < NGrids; n++)
    {
        const RealMatrix&& Vn = PES[HamiltonianType](GridCoordinate[n]);
        for (int m = 0; m < NumPES; m++)
        {
            for (int mm = 0; mm < NumPES; mm++)
            {
                Hamiltonian(m * NGrids + n, mm * NGrids + n) += Vn(m, mm);
            }
        }
    }
    // 2. -hbar^2/2M*(d^2)_(mm'),if n=n'
    // or +hbar^2/M*d(mm'n)(-1)^(n+n')/(n'-n)/dx, if n!=n'
    // here, d2n=D^2(Rn),sum_k(d(mkn)d(km'n))=D^2(Rn)[m][m']
    if (HamiltonianType != Diabatic)
    {
        for (int n = 0; n < NGrids; n++)
        {
            const RealMatrix&& dn = NAC[HamiltonianType](GridCoordinate[n]);
            RealMatrix d2n(NumPES);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, NumPES, NumPES, NumPES, 1.0, dn.data(), NumPES, dn.data(), NumPES, 0.0, d2n.data(), NumPES);
            for (int m = 0; m < NumPES; m++)
            {
                for (int nn = 0; nn < NGrids; nn++)
                {
                    if (nn == n)
                    {
                        for (int mm = 0; mm < NumPES; mm++)
                        {
                            Hamiltonian(m * NGrids + n, mm * NGrids + nn) -= hbar * hbar / 2 / mass * d2n(m, mm);
                        }
                    }
                    else
                    {
                        for (int mm = 0; mm < NumPES; mm++)
                        {
                            Hamiltonian(m * NGrids + n, mm * NGrids + nn) += pow_minus_one(nn - n) * hbar * hbar / (nn - n) / dx / mass * dn(m, mm);
                        }
                    }
                }
            }
        }
    }
    // 3. d2/dx2 (over all pes)
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


    // evolve; if H is hermitian, diagonal; otherwise, RK4
    if (Absorbed != true && HamiltonianType == Diabatic)
    {
        // diagonalize H
        ComplexMatrix EigVec(Hamiltonian);
        double* EigVal = new double[dim];
        if (LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', dim, reinterpret_cast<MKL_Complex16*>(EigVec.data()), dim, EigVal) > 0)
        {
            cerr << "FAILING DIAGONALIZE DIABATIC HAMILTONIAN IN DYNAMICS" << endl;
            exit(300);
        }
        // H(diag)=EigVal=[EigVec]^T*H*[EigVec]
        Complex* DiagH = new Complex[dim];
        real_to_complex(EigVal, DiagH, dim);
        // iHdt = -iH(diag)t, exp_iHdt = exp(-iH(diag)t)
        Complex* iHdt = new Complex[dim];
        Complex* exp_iHdt = new Complex[dim];
        // psi_t=psi(t)=exp(-iHt)*psi(0)
        Complex* psi_t = new Complex[dim];
        // EigPropa is an intermediate, C*exp(-iH(diag)dt)
        // original representation exp(-iHdt)=PropaEig*C^dagger
        Complex* PropaEig = new Complex[dim * dim];
        for (int iStep = 0; iStep <= TotalStep; iStep += PsiOutputStep)
        {
            // calculate exp(-iH(diag)dt)
            Complex propalpha(0, -iStep * dt); 
            memset(iHdt, 0, dim * sizeof(Complex));
            cblas_zaxpy(dim, &propalpha, DiagH, 1, iHdt, 1);
            vmzExp(dim, reinterpret_cast<const MKL_Complex16*>(iHdt), reinterpret_cast<MKL_Complex16*>(exp_iHdt), mode);
            // reconstruct the H matrix, and exp(-iHdt)=C*exp(-iH(diag)dt)*C^T
            ComplexMatrix Propagator(dim);
            for (int i = 0; i < dim; i++)
            {
                Propagator(i, i) = exp_iHdt[i];
            }
            cblas_zsymm(CblasRowMajor, CblasRight, CblasUpper, dim, dim, &Alpha, Propagator.data(), dim, EigVec.data(), dim, &Beta, PropaEig, dim);
            cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans, dim, dim, dim, &Alpha, PropaEig, dim, EigVec.data(), dim, &Beta, Propagator.data(), dim);
            // calculate psi_t=psi(t)=exp(-iHt)*psi(0)
            cblas_zgemv(CblasRowMajor, CblasNoTrans, dim, dim, &Alpha, Propagator.data(), dim, Psi, 1, &Beta, psi_t, 1);
            // print
            PsiOutput << iStep * dt;
            for (int i = 0; i < dim; i++)
            {
                PsiOutput << ' ' << psi_t[i].real() << ' ' << psi_t[i].imag();
            }
            PsiOutput << endl;
            // check if calculating phase space distribution
            if (iStep % PhaseOutputStep == 0)
            {
                PhaseOutput << iStep * dt;
                // Wigner Transformation: P(x,p)=int{dy*exp(2ipy/hbar)<x-y|rho|x+y>}/(pi*hbar)
                // the interval of p is pi*hbar/dx*[-1,1), dp=2*pi*hbar/(xmax-xmin)
                // loop over p first
                for (int i = 0; i < NGrids; i++)
                {
                    const double p = (i - NGrids / 2) * 2 * pi * hbar / (xmax - xmin + (Absorbed == true ? 2 * AbsorbingRegionLength : 0));
                    // loop over x
                    for (int j = 0; j < NGrids; j++)
                    {
                        // do the numerical integral and output
                        Complex integral;
                        for (int k = max(-j, j + 1 - NGrids); k <= min(j, NGrids - 1 - j); k++)
                        {
                            integral += exp(2.0i * p * GridCoordinate[k] / hbar) * conj(Psi[j + k]) * Psi[j - k];
                        }
                        PhaseOutput << ' ' << integral.real() / pi / hbar;
                    }
                }
                PhaseOutput << endl;
            }
        }
        delete[] PropaEig;
        delete[] psi_t;
        delete[] exp_iHdt;
        delete[] iHdt;
        delete[] DiagH;
        delete[] EigVal;
    }

    // end. free the memory, close the files.
    delete[] Psi;
    delete[] GridCoordinate;
    PhaseOutput.close();
    PsiOutput.close();
	return 0;
}
