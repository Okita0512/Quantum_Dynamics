from scipy import integrate

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

# DVR parameters
Ngrid = 1001
L = 2.0
R = np.linspace(- L, L, Ngrid)
dx = R[1] - R[0]

# Model parameters
"""
the model Arkajit used, see orginal paper: JCP 140, 174105 (2014)
"""
m_s = 1836
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

# ==============================================================================================
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
eigenvec[:, 0] = - eigenvec[:, 0]       # be careful here, it is inversed initially 

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

"""
    Hsys = Hm + (RE1 + RE2) * R^2
        matter    reorganization energy
"""

def get_Hs(NStates, lambda_1, RE_c):

    hams = np.zeros((NStates, NStates), dtype = complex)

    R_1 = get_R(NStates)
    R_2 = np.dot(R_1, R_1)

    # reorganization energy for the Debye bath
    """
    RE = (1/pi) \int_0^{+\infty} dw [J(w) / w]. For the Debye bath, it is just lambda_1
    """
    RE = lambda_1 + RE_c

    for i in range(NStates):
        for j in range(NStates):

            hams[i, j] = eigenvalue[i] * delta(i, j) + RE * R_2[i, j]

    return hams

def get_Qs1(NStates):  # R = R_ij |v_i >< v_j|
    return get_R(NStates)

def get_Qs2(NStates):  # mu = R
    return get_R(NStates)

def get_overlap(state1, state2):

    P = 0.0
    center = int(len(eigenvec[:, 0])/2)

    for k in range(0, center):
        P += eigenvec[k, state1].conjugate() * eigenvec[k, state2]
    
    P += eigenvec[center, state1].conjugate() * eigenvec[center, state2] / 2.0

    return P

def get_rho0(NStates, temp):

    beta = 1.0 / temp
    rho0 = np.zeros((NStates, NStates), dtype = complex)

    # obtain the system subspace partition function
#    Z_s = 0
#    for j in range(NStates):
#        Z_s += 0.5 * np.exp(- beta * eigenvalue[j])

    # thermalized DM
#    for i in range(NStates):
#        for j in range(NStates):
#
#            rho0[i, j] = np.exp(- 0.5 * beta * (eigenvalue[i] + eigenvalue[j])) * get_overlap(i, j)
#
#    return rho0 / Z_s

    # product DM

    for i in range(NStates):
        for j in range(NStates):

            rho0[i, j] = 0.5 * delta(i, 0) * delta(j, 0) + 0.5 * delta(i, 1) * delta(j, 1) - 0.5 * delta(i, 0) * delta(j, 1) - 0.5 * delta(i, 1) * delta(j, 0)
    
    return rho0

def λc(tau, ωc, gamma, temp):

    beta = 1. / temp
    lr = 1. / tau     # cavity loss rate, defined as 1 / relaxation time

    return lr * (1.0 - np.exp(- beta * ωc)) * ( gamma**2 + ωc**2 ) / (4 * ωc * gamma)

def gen_jw(w, omega_c, lam, Gamma):
    return 2 * lam * omega_c**2 * Gamma * w / ((w**2 - omega_c**2)**2 + (Gamma * w)**2)

# ==============================================================================================
#                                    Summary of parameters     
# ==============================================================================================
class parameters:

    # ===== DEOM propagation scheme =====
    dt = 0.025 * fs_to_au
    t = 10000 * fs_to_au    # plateau time as 20ps for HEOM
    nt = int(t / dt)
    nskip = 100

    lmax = 20
    nmax = 1000000
    ferr = 1.0e-07

    # ===== number of system states =====
    NStates = 10                        # total number of matter states

    # ===== Cavity parameters =====
    omega_c = 1000 * cm_to_au         # cavity frequency. Note that in this model, the energy gap is around 1140 cm^-1
    eta_c = 0.025                  # light-matter-coupling strength. Set as 0 when cavity is turned off

    # ===== Drude-Lorentz model =====
    temp = 300 / au_to_K                             # temperature
    nmod = 2                                         # number of dissipation modes

    # Bath I parameters, Drude-Lorentz model
    gamma_1   = 200 * cm_to_au                      # bath characteristic frequency
    ratio = 0.1                                     # the value of etas / omega_b, tune it from 0.02 to 2.0
    lambda_1 = ratio * m_s * omega_b * gamma_1 / 2        # reorganization energy

    # PSD scheme
    pade    = 1                            # 1 for [N-1/N], 2 for [N/N], 3 for [N+1/N]
    npsd    = 3                            # number of Pade terms

    # Bath II parameters, Brownian Oscillator
    tau_c = 1000 * fs_to_au                           # bath relaxation time
    gamma_2 = 1. / tau_c                        # bath characteristic frequency  
    lambda_2 = eta_c**2 * omega_c        # reorganization energy       

    # ===== Get the subspace information ===== 
    eigenvalue = eigenvalue
    eigenvec = eigenvec

    # ===== Build the bath-free Hamiltonian, dissipation operators, and initial DM in the subspace =====
    
    Qs1 = get_Qs1(NStates)
    Qs2 = get_Qs2(NStates)
    rho0 = get_rho0(NStates, temp)

def Jw_w(x):
    omega_c, lambda_2, gamma_2 = parameters.omega_c, parameters.lambda_2, parameters.gamma_2
    return gen_jw(x, omega_c, lambda_2, gamma_2) / (np.pi * x)

# print((parameters.eigenvalue[3] - parameters.eigenvalue[0]) / cm_to_au)

# ==============================================================================================
#                                         Main Program     
# ==============================================================================================

if __name__ == '__main__':

    with open('default.json') as f:
        ini = json.load(f)

    # passing parameters
    # cavity
    omega_c = parameters.omega_c
    # bath
    temp = parameters.temp
    nmod = parameters.nmod
    lambda_1 = parameters.lambda_1
    gamma_1 = parameters.gamma_1
    lambda_2 = parameters.lambda_2
    gamma_2 = parameters.gamma_2
    pade = parameters.pade
    npsd = parameters.npsd
    w = np.linspace(0.00001 * omega_c, 10 * omega_c, 10000000)
    y = Jw_w(w)
    RE_c = integrate.trapz(y, w)
    print(RE_c)
    # system
    NStates = parameters.NStates
    hams = get_Hs(NStates, lambda_1, RE_c)
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

	# bath PSD
    ini['bath']['temp'] = temp
    ini['bath']['nmod'] = nmod
    ini['bath']['pade'] = pade
    ini['bath']['npsd'] = npsd
    ini['bath']['jomg'] = [{"jdru":[(lambda_1, gamma_1)]}] 
                                                                               
    jomg = ini['bath']['jomg']
    nind = 0
    for m in range(nmod - 1):       # one mode is treated by PFD
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
                                                                               
    etal_1, etar_1, etaa_1, expn_1, delr_1 = generate (temp, npsd, pade, jomg)

	# bath II with PSD
    ini['bath']['temp'] = temp                                                  
    ini['bath']['nmod'] = nmod
    ini['bath']['pade'] = pade
    ini['bath']['npsd'] = npsd + 1
    ini['bath']['jomg'] = [{"jsdr":[(lambda_2, omega_c, gamma_2)]}] 
                                                                                
    jomg = ini['bath']['jomg']
    mode = np.zeros((nind + 2 + npsd + 1), dtype = int)
    for i in range(nind, nind + npsd + 3):
        mode[i] = 1

    etal_2, etar_2, etaa_2, expn_2, delr_2 = generate (temp, npsd + 1, pade, jomg)

    delr = np.append(delr_1, delr_2)
    etal = np.append(etal_1, etal_2)
    etar = np.append(etar_1, etar_2)
    etaa = np.append(etaa_1, etaa_2)
    expn = np.append(expn_1, expn_2)

    arma.arma_write(mode, 'inp_mode.mat')
    arma.arma_write(delr, 'inp_delr.mat')
    arma.arma_write(etal, 'inp_etal.mat')
    arma.arma_write(etar, 'inp_etar.mat')
    arma.arma_write(etaa, 'inp_etaa.mat')
    arma.arma_write(expn, 'inp_expn.mat')

    # two dissipation modes
    qmds = np.zeros((nmod, NStates, NStates), dtype = complex)
    qmds[0,:,:] = Qs1                           # the electron-phonon interaction
    qmds[1,:,:] = Qs2                           # the effective spectral density for the cavity

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
    arma.arma_write(sdip,'inp_sdip.mat')

    pdip = np.zeros((nmod,2,2),dtype=float)
    pdip[0,0,1] = pdip[0,1,0] = 1.0
    arma.arma_write(pdip,'inp_pdip.mat')

    bdip = np.zeros(3,dtype=complex)
#    bdip[0]=-complex(5.00000000e-01,8.66025404e-01)
#    bdip[1]=-complex(5.00000000e-01,-8.66025404e-01)
#    bdip[2]=-complex(7.74596669e+00,0.00000000e+00)
    arma.arma_write(bdip,'inp_bdip.mat')

    with open('input.json','w') as f:
        json.dump(jsonInit,f,indent=4) 
