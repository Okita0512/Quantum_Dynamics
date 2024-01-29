import json
import numpy as np
import armadillo as arma
from bath_gen_Drude_PSD import generate

# ==============================================================================================
#                                       Global Parameters     
# ==============================================================================================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# ================= DVR calculation of the double-well potential ====================

Ngrid = 1001
L = 2.0
R = np.linspace(- L, L, Ngrid)
dx = R[1] - R[0]

m_s = 1836                                      # mass, 1836 for proton
omega_b = 1000 * cm_to_au
E_b = 2120 * cm_to_au

def kinetic(Ngrid, M, dx):

    A = np.zeros((Ngrid, Ngrid), dtype=float)

    for i in range(Ngrid):
        A[i, i] = np.pi**2 / 3
        for j in range(1, Ngrid - i):
            A[i, i+j] = 2 * (-1)**j / j**2
            A[i+j, i] = A[i, i+j]

    A = A / (2 * M * dx**2)

    return A

def potential(Ngrid, R, m_s, omega_b, E_b):
    
    def V(R):
        return - (m_s * omega_b**2 / 2) * R**2 + (m_s**2 * omega_b**4 / (16 * E_b)) * R**4
        
    B = np.zeros((Ngrid, Ngrid), dtype=float)

    for i in range(Ngrid):
        B[i, i] = V(R[i])

    return B

def diagonalization(Ngrid, dx, R, m_s, omega_b, E_b):

    H = kinetic(Ngrid, m_s, dx) + potential(Ngrid, R, m_s, omega_b, E_b)
    eigenvalue, eigenvec = np.linalg.eig(H)

    return eigenvalue, eigenvec

# ============================================================================
# get the vibrational eigen states and sort ascendingly
eigenvalue, eigenvec = diagonalization(Ngrid, dx, R, m_s, omega_b, E_b)
eigenvec = eigenvec
ordered_list = sorted(range(len(eigenvalue)), key=lambda k: eigenvalue[k])

temp1 = np.zeros((len(eigenvec[:, 0]), len(eigenvec[0, :])), dtype = complex)
temp2 = np.zeros((len(eigenvalue)), dtype=float)

for count in range(len(eigenvalue)):
    temp1[:, count] = eigenvec[:, ordered_list[count]]
    temp2[count] = eigenvalue[ordered_list[count]]
eigenvec = temp1
eigenvalue = temp2
eigenvec[:, 0] = - eigenvec[:, 0]

del temp1
del temp2
del ordered_list

# ==============================================================================================
#                                       Auxiliary functions     
# ==============================================================================================

def delta(m, n):
    return 1 if m == n else 0

def get_R(NStates):

    R_s = np.zeros((NStates, NStates), dtype=complex)

    for i in range(NStates):
        for j in range(NStates):
            for k in range(len(eigenvec[:, i])):
                R_s[i,j] += eigenvec[k, i].conjugate() * R[k] * eigenvec[k, j]
    
    return np.real(R_s)

def get_Hs(n_vib, n_ph, omega_c, eta_c, lambda_1, lambda_2):

    nfock = n_vib * n_ph
    hams = np.zeros((nfock, nfock), dtype = complex)

    R_1 = get_R(n_vib)
    R_2 = np.matmul(R_1, R_1)

    RE_1 = lambda_1 + omega_c * eta_c**2
    RE_2 = lambda_2

    for i in range(n_vib):
        for j in range(n_vib):
            for m in range(n_ph):
                for n in range(n_ph):
                    
                    hams[n_vib * m + i, n_vib * n + j] = ( (eigenvalue[i] + (m + 0.5) * omega_c) * delta(i, j) * delta(m, n)

                        + omega_c * eta_c * R_1[i, j] * (np.sqrt(n + 1) * delta(m, n + 1) + np.sqrt(n) * delta(m, n - 1)) 

                        + RE_1 * R_2[i, j] * delta(m, n) + RE_2 * delta(i, j) * (np.sqrt((m + 1) * (m + 2)) * delta(n, m + 2) 
                        + (2 * m + 1) * delta(n, m) + np.sqrt(m * (m - 1)) * delta(n, m - 2) )

                        )

    return hams

def get_Qs1(n_vib, n_ph):  # R

    nfock = n_vib * n_ph
    Qs1 = np.zeros((nfock, nfock), dtype = complex)

    R_1 = get_R(n_vib)

    for i in range(n_vib):
        for j in range(n_vib):
            for m in range(n_ph):
                for n in range(n_ph):

                    Qs1[n_vib * m + i, n_vib * n + j] = R_1[i, j] * delta(m, n)

    return Qs1

def get_Qs2(n_vib, n_ph):  # a + a^+

    nfock = n_vib * n_ph
    Qs2 = np.zeros((nfock, nfock), dtype = complex)

    for i in range(n_vib):
        for j in range(n_vib):
            for m in range(n_ph):
                for n in range(n_ph):

                    Qs2[n_vib * m + i, n_vib * n + j] = delta(i, j) * ( np.sqrt(n + 1) * delta(m, n + 1) + np.sqrt(n) * delta(m, n - 1) )

    return Qs2

def get_rho0(n_vib, n_ph):

    nfock = n_vib * n_ph
    rho0 = np.zeros((nfock, nfock), dtype = complex)

    for i in range(nfock):
        for j in range(nfock):

            rho0[i, j] = 0.5 * delta(i, 0) * delta(j, 0) + 0.5 * delta(i, 1) * delta(j, 1) - 0.5 * delta(i, 0) * delta(j, 1) - 0.5 * delta(i, 1) * delta(j, 0)
    
    return rho0

def λc(tau, ωc, gamma, temp):   # defined according to my paper. Arkajit's 1000 fs is equivalent to my 2000 fs
    return (1.0 - np.exp(- ωc / temp)) * (gamma**2 + ωc**2 ) / (4 * tau * ωc * gamma)

# ==============================================================================================
#                                    Summary of parameters     
# ==============================================================================================

class parameters:

    # ===== DEOM propagation scheme =====
    dt = 0.025 * fs_to_au
    t = 2000 * fs_to_au    # plateau time as 20ps for HEOM
    nt = int(t / dt)
    nskip = 100

    lmax = 10
    nmax = 1000000
    ferr = 1.0e-07

    # ===== number of system states =====
    n_vib = 4
    n_ph = 5
    nfock = n_vib * n_ph       # vibrational * photonic

    ## cavity parameters
    omega_c = 1172 * cm_to_au   # cavity frequency. Note that in this model, the energy gap is around 1185.8 cm^-1
    eta_c = 0.05                 # light-matter-coupling strength

    # ===== Drude-Lorentz model =====
    temp    = 300 / au_to_K                             # temperature
    nmod    = 2                                         # number of dissipation modes

    # Bath I parameters, Drude-Lorentz model
    gamma_1   = 200 * cm_to_au                      # bath characteristic frequency
    ratio = 2.0                                     # the value of etas / omega_b, tune it from 0.02 to 2.0
    lambda_1 = ratio * m_s * omega_b * gamma_1 / 2        # reorganization energy

    # PSD scheme
    pade    = 1                            # 1 for [N-1/N], 2 for [N/N], 3 for [N+1/N]
    npsd    = 3                            # number of Pade terms

    # Bath II parameters, Brownian Oscillator
    gamma_2 = 1e3 * cm_to_au                        # bath characteristic frequency  
    tau_c = 200 * fs_to_au                           # bath relaxation time
    lambda_2 = λc(tau_c, omega_c, gamma_2, temp)     # reorganization energy       

    # ===== Get the subspace information ===== 
    eigenvalue = eigenvalue
    eigenvec = eigenvec

    # ===== Build the bath-free Hamiltonian, dissipation operators, and initial DM in the subspace =====
    
    Qs1 = get_Qs1(n_vib, n_ph)
    Qs2 = get_Qs2(n_vib, n_ph)
    rho0 = get_rho0(n_vib, n_ph)
    hams = get_Hs(n_vib, n_ph, omega_c, eta_c, lambda_1, lambda_2)
    
# ==============================================================================================
#                                         Main Program     
# ==============================================================================================

if __name__ == '__main__':

    with open('default.json') as f:
        ini = json.load(f)

    # passing parameters
    # cavity
    omega_c = parameters.omega_c
    eta_c = parameters.eta_c
    # bath
    temp = parameters.temp
    nmod = parameters.nmod
    lambda_1 = parameters.lambda_1
    gamma_1 = parameters.gamma_1
    lambda_2 = parameters.lambda_2
    gamma_2 = parameters.gamma_2
    pade = parameters.pade
    npsd = parameters.npsd
    # system
    nfock = parameters.nfock
    hams = parameters.hams
    rho0 = parameters.rho0
    # system-bath
    Qs1 = parameters.Qs1
    Qs2 = parameters.Qs2
    # DEOM
    dt = parameters.dt
    nt = parameters.nt
    nskip = parameters.nskip
    lmax = parameters.lmax
    nmax = parameters.nmax
    ferr = parameters.ferr

# ==============================================================================================================================
    # hidx
    ini['hidx']['trun'] = 0
    ini['hidx']['lmax'] = lmax
    ini['hidx']['nmax'] = nmax
    ini['hidx']['ferr'] = ferr

	# bath
    ini['bath']['temp'] = temp
    ini['bath']['nmod'] = nmod
    ini['bath']['jomg'] = [{"jdru":[(lambda_1, gamma_1)]}, {"jdru":[(lambda_2, gamma_2)]}] # homogeneous Drude-Lorentz
    ini['bath']['pade'] = pade
    ini['bath']['npsd'] = npsd

    jomg = ini['bath']['jomg']
    nind = 0
    for m in range(nmod):       # one mode is treated by PFD
        try:
            ndru = len(jomg[m]['jdru'])
        except:
            ndru = 0
        try:
            nsdr = len(jomg[m]['jsdr'])
        except:
            nsdr = 0
        nper = ndru + 2 * nsdr + npsd
        nind += nper

    etal, etar, etaa, expn, delr = generate (temp, npsd, pade, jomg)
    
    mode = np.zeros((nind), dtype = int)
    for i in range(npsd + 1, nind):
        mode[i] = 1

    arma.arma_write(mode, 'inp_mode.mat')
    arma.arma_write(delr, 'inp_delr.mat')
    arma.arma_write(etal, 'inp_etal.mat')
    arma.arma_write(etar, 'inp_etar.mat')
    arma.arma_write(etaa, 'inp_etaa.mat')
    arma.arma_write(expn, 'inp_expn.mat')

    # two dissipation modes, one is electron-phonon interaction, the other is cavity loss
    qmds = np.zeros((nmod, nfock, nfock), dtype = complex)
    qmds[0,:,:] = Qs1       # the electron-phonon interaction, H_sb = R ⊗ \sum_j c_j x_j, described by J(w). Q_s = R
    qmds[1,:,:] = Qs2       # the cavity loss, H'_sb = (a + a^+) ⊗ \sum_j c_j x_j, described by J'(w). Q'_s = a + a^+

    arma.arma_write (hams,ini['syst']['hamsFile'])
    arma.arma_write (qmds,ini['syst']['qmdsFile'])
    arma.arma_write (rho0,'inp_rho0.mat')

    jsonInit = {"deom":ini,
                "rhot":{
                    "dt": dt,
                    "nt": nt,
                    "nk": nskip,
					"xpflag": 1,
					"staticErr": 0,
                    "rho0File": "inp_rho0.mat",
                    "sdipFile": "inp_sdip.mat",
                    "pdipFile": "inp_pdip.mat",
					"bdipFile": "inp_bdip.mat"
                },
            }

# ==============================================================================================================================
# ==============================================================================================================================

    # dipoles
    sdip = np.zeros((2,2),dtype=float)
    arma.arma_write (sdip,'inp_sdip.mat')

    pdip = np.zeros((nmod,2,2),dtype=float)
    pdip[0,0,1] = pdip[0,1,0] = 1.0
    arma.arma_write (pdip,'inp_pdip.mat')

    bdip = np.zeros(3,dtype=complex)
#    bdip[0]=-complex(5.00000000e-01,8.66025404e-01)
#    bdip[1]=-complex(5.00000000e-01,-8.66025404e-01)
#    bdip[2]=-complex(7.74596669e+00,0.00000000e+00)
    arma.arma_write (bdip,'inp_bdip.mat')

    with open('input.json','w') as f:
        json.dump(jsonInit,f,indent=4) 
