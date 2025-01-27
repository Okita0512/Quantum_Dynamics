import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
from matplotlib.pyplot import MultipleLocator, tick_params

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

# Model parameters: DW2
m_s = 1836
omega_b = 1000 * cm_to_au
E_b = 2120 * cm_to_au

# ===== Cavity parameters =====
omega_c = 1180 * cm_to_au         # cavity frequency. Note that in this model, the energy gap is around 1140 cm^-1
eta_c = 0.005                     # light-matter-coupling strength. Set as 0 when cavity is turned off

# ===== Drude-Lorentz model =====
temp    = 300 / au_to_K                             # temperature

# Bath I parameters, Drude-Lorentz model
gamma_1   = 200 * cm_to_au                      # bath characteristic frequency
ratio = 0.1                                     # the value of etas / omega_b, tune it from 0.02 to 2.0
lambda_1 = ratio * m_s * omega_b * gamma_1 / 2        # reorganization energy

def λc(tau, ωc, gamma, temp):

    beta = 1. / temp
    lr = 1. / tau     # cavity loss rate, defined as 1 / relaxation time

    return lr * (1.0 - np.exp(- beta * ωc)) * ( gamma**2 + ωc**2 ) / (4 * ωc * gamma)

# Bath II parameters, Brownian Oscillator
gamma_2 = 1000 * cm_to_au                        # bath characteristic frequency  
tau_c = 1000 * fs_to_au                           # bath relaxation time
lambda_2 = λc(tau_c, omega_c, gamma_2, temp)     # reorganization energy       

# ==============================================================================================
#                                       Auxiliary functions     
# ==============================================================================================
def kinetic(Ngrid, M, dx):

    A = np.zeros((Ngrid, Ngrid), dtype=float)

    for i in range(Ngrid):
        A[i, i] = np.pi**2 / 3
        for j in range(1, Ngrid - i):
            A[i, i+j] = 2 * (-1)**j / j**2
            A[i+j, i] = A[i, i+j]

    A = A / (2 * M * dx**2)

    return A

def V(R):
    return - (m_s * omega_b**2 / 2) * R**2 + (m_s**2 * omega_b**4 / (16 * E_b)) * R**4

def potential(Ngrid, R):

    B = np.zeros((Ngrid, Ngrid), dtype=float)

    for i in range(Ngrid):
        B[i, i] = V(R[i])

    return B

def diagonalization(Ngrid, dx, R, m_s):

    H = kinetic(Ngrid, m_s, dx) + potential(Ngrid, R)
    eigenvalue, eigenvec = np.linalg.eig(H)

    return eigenvalue, eigenvec

def gen_jw(w, omega_c, eta_c, lam, gamma):

    J0 = (2 * lam * gamma * w / (w**2 + gamma**2)) * (2 * omega_c)
    zeta = np.sqrt(2 / omega_c) * eta_c

    return (omega_c**4 * zeta**2 * J0) / ((omega_c**2 - w**2 + (w * J0 / gamma))**2 + (J0)**2) * np.heaviside(w, 0)

def Drude(w, lam, gam):
    return 2 * lam * gam * w / (w**2 + gam**2) * np.heaviside(w, 0)

# ==============================================================================================
#                                       Plotting Fig 1a     
# ==============================================================================================

lw = 3.0
unitlen = 10
legendsize = 36         # size for legend
font_legend = {'weight': 'roman', 'size': 19}

fig, ax = plt.subplots(figsize=(2 * unitlen, 0.9 * unitlen), dpi = 512)
# fig.subplots_adjust(hspace = 0.25, wspace = 0.0)

plt.subplot(1,2,1)

# get the vibrational eigen states and sort ascendingly
eigenvalue, eigenvec = diagonalization(Ngrid, dx, R, m_s)
eigenvec = eigenvec
ordered_list = sorted(range(len(eigenvalue)), key=lambda k: eigenvalue[k])

temp1 = np.zeros((len(eigenvec[:, 0]), len(eigenvec[0, :])), dtype = complex)
temp2 = np.zeros((len(eigenvalue)), dtype=float)

for count in range(len(eigenvalue)):
    temp1[:, count] = eigenvec[:, ordered_list[count]]
    temp2[count] = eigenvalue[ordered_list[count]]
eigenvec = np.real(temp1)
eigenvalue = np.real(temp2)

eigenvec[:, 1] = - eigenvec[:, 1] 

# ======== diabatization ===============
ave1 = (eigenvalue[0] + eigenvalue[1]) / 2
eigenvalue[0] = ave1
eigenvalue[1] = ave1

ave2 = (eigenvalue[2] + eigenvalue[3]) / 2
eigenvalue[2] = ave2
eigenvalue[3] = ave2

vL1 = (eigenvec[:, 0] + eigenvec[:, 1]) / 2
vR1 = (eigenvec[:, 0] - eigenvec[:, 1]) / 2
eigenvec[:, 0] = vL1
eigenvec[:, 1] = vR1

vL2 = (eigenvec[:, 2] + eigenvec[:, 3]) / 2
vR2 = (eigenvec[:, 2] - eigenvec[:, 3]) / 2
eigenvec[:, 2] = vL2
eigenvec[:, 3] = vR2
# =======================================

del temp1
del temp2
del ordered_list

# check the correctness
ratio_1 = cm_to_au
ratio_2 = 0.0002

color1 = 'silver'
color2 = 'gray'
color3 = '#33d572' # green, |3>
color4 = '#e7513e' # red, |2>
color5 = '#378ada' # blue, |1>
color6 = '#f1a51e' # orange, |0>

# plt.plot(R, [(eigenvalue[5] - eigenvalue[0]) / ratio_1] * len(R), '-', linewidth = 1.0, color = color1)
# plt.plot(R, [(eigenvalue[4] - eigenvalue[0]) / ratio_1] * len(R), '-', linewidth = 1.0, color = color2)
plt.plot(R, [(eigenvalue[3] - eigenvalue[0]) / ratio_1] * len(R), '-', linewidth = 1.0, color = color3)
plt.plot(R, [(eigenvalue[2] - eigenvalue[0]) / ratio_1] * len(R), '-', linewidth = 1.0, color = color4)
plt.plot(R, [(eigenvalue[1] - eigenvalue[0]) / ratio_1] * len(R), '-', linewidth = 1.0, color = color5)
plt.plot(R, [(eigenvalue[0] - eigenvalue[0]) / ratio_1] * len(R), '-', linewidth = 1.0, color = color6)

plt.plot(R, (V(R) - eigenvalue[0]) / ratio_1, linewidth = lw, color = 'black')
# plt.plot(R, (eigenvalue[5] - eigenvalue[0]) / ratio_1 + eigenvec[:, 5] / ratio_2, linewidth = lw, color = color1, label = r'$\|\nu_5\rangle$')
# plt.plot(R, (eigenvalue[4] - eigenvalue[0]) / ratio_1 + eigenvec[:, 4] / ratio_2, linewidth = lw, color = color2, label = r'$\|\nu_4\rangle$')
plt.plot(R, (eigenvalue[2] - eigenvalue[0]) / ratio_1 + eigenvec[:, 2] / ratio_2, linewidth = lw, color = color4, label = r"$\|\nu'_\mathrm{R}\rangle$")
plt.plot(R, (eigenvalue[3] - eigenvalue[0]) / ratio_1 + eigenvec[:, 3] / ratio_2, linewidth = lw, color = color3, label = r"$\|\nu'_\mathrm{L}\rangle$")
plt.plot(R, (eigenvalue[1] - eigenvalue[0]) / ratio_1 + eigenvec[:, 1] / ratio_2, linewidth = lw, color = color5, label = r'$\|\nu_\mathrm{R}\rangle$')
plt.plot(R, (eigenvalue[0] - eigenvalue[0]) / ratio_1 + eigenvec[:, 0] / ratio_2, linewidth = lw, color = color6, label = r'$\|\nu_\mathrm{L}\rangle$')

# plt.fill_between(R,(eigenvalue[5] - eigenvalue[0]) / ratio_1, (eigenvalue[5] - eigenvalue[0]) / ratio_1 + eigenvec[:, 5] / ratio_2, color = color1, alpha = .6)
# plt.fill_between(R,(eigenvalue[4] - eigenvalue[0]) / ratio_1, (eigenvalue[4] - eigenvalue[0]) / ratio_1 + eigenvec[:, 4] / ratio_2, color = color2, alpha = .6)
plt.fill_between(R,(eigenvalue[3] - eigenvalue[0]) / ratio_1, (eigenvalue[3] - eigenvalue[0]) / ratio_1 + eigenvec[:, 3] / ratio_2, color = color3, alpha = .6)
plt.fill_between(R,(eigenvalue[2] - eigenvalue[0]) / ratio_1, (eigenvalue[2] - eigenvalue[0]) / ratio_1 + eigenvec[:, 2] / ratio_2, color = color4, alpha = .6)
plt.fill_between(R,(eigenvalue[1] - eigenvalue[0]) / ratio_1, (eigenvalue[1] - eigenvalue[0]) / ratio_1 + eigenvec[:, 1] / ratio_2, color = color5, alpha = .6)
plt.fill_between(R,(eigenvalue[0] - eigenvalue[0]) / ratio_1, (eigenvalue[0] - eigenvalue[0]) / ratio_1 + eigenvec[:, 0] / ratio_2, color = color6, alpha = .6)

# scale for major and minor locator
x_major_locator = MultipleLocator(1)
x_minor_locator = MultipleLocator(0.5)
y_major_locator = MultipleLocator(500)
y_minor_locator = MultipleLocator(100)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 8, labelsize = 10)
ax.tick_params(which = 'minor', length = 4)

# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

x0 = 1.8
y1, y2 = - 800, 2500

plt.tick_params(labelsize = 20, which = 'both', direction = 'in')
plt.xlim(- x0, x0)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 8)
ax2.tick_params(which = 'minor', length = 4)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel(r'$\mathrm{R\ (a.u.)}$', size = 24)
ax.set_ylabel(r'$\omega = E - E_0\ (\mathrm{cm}^{-1})$', size = 24)
ax.legend(loc = 'upper center', frameon = False, prop = font_legend)
plt.legend(title = '(a)   ', frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                       Plotting Fig 1b     
# ==============================================================================================

plt.subplot(1,2,2)

lw = 4.0
font_legend = {'weight': 'roman', 'size': 28}

w1 = np.linspace(- omega_c, 3 * omega_c, 10000)
w2 = np.linspace(- omega_c, 3 * omega_c, 10000)

ratio1 = 1e-05
ratio2 = 1e-04

# scale for major and minor locator
x_major_locator = MultipleLocator(2)
x_minor_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(500)
y_minor_locator = MultipleLocator(100)

# # lower x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 8)
ax.tick_params(which = 'minor', length = 4)

j_0 = 7.0

plt.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 20)
plt.tick_params(axis = 'y', which = 'both', direction = 'in', labelsize = 0)
plt.xlim(- 0.1, j_0)
plt.ylim(y1, y2)

# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

ax.set_xlabel(r'$\mathcal{J}\mathrm{_{diss}}(\omega)\ \mathrm{Intensity}\ (\times 10^{-4}\ \mathrm{a.u.})$', size = 24)

# # upper x-axis
ax2 = ax.twiny()
ax2.xaxis.set_major_locator(x_major_locator)
ax2.xaxis.set_minor_locator(x_minor_locator)
ax2.tick_params(which = 'major', length = 8)
ax2.tick_params(which = 'minor', length = 4)

# x2_label = ax2.get_xticklabels()
# [x2_label_temp.set_fontname('Times New Roman') for x2_label_temp in x2_label]

plt.tick_params(labelsize = 0, which = 'both', direction = 'in')
plt.xlim(- 0.1, j_0)
# ax2.set_xlabel(r'$\mathcal{J}\mathrm{_{eff}}(\omega)\ \mathrm{Intensity}\ (\times 10^{-5}\ \mathrm{a.u.})$', size = 18)

# RHS y-axis
ax3 = ax.twinx()
ax3.yaxis.set_major_locator(y_major_locator)
ax3.yaxis.set_minor_locator(y_minor_locator)
ax3.tick_params(which = 'major', length = 8)
ax3.tick_params(which = 'minor', length = 4)
plt.tick_params(labelsize = 0, which = 'both', direction = 'in')
plt.ylim(y1, y2)

ax.plot(Drude(w2, lambda_1, gamma_1) / ratio2, w2 / cm_to_au, linewidth = lw, color = 'blue', label = r"$\mathcal{J}\mathrm{_{diss}}(\omega)$")
# ax2.plot(gen_jw(w1, omega_c, eta_c, λc(tau_c, omega_c, gamma_2, temp), gamma_2) / ratio1, w1 / cm_to_au, linewidth = lw, color = 'red', label = r"$\mathcal{J}\mathrm{_{eff}}(\omega)$")

ax.legend(loc = 'lower center', frameon = False, prop = font_legend)
# ax2.legend(loc = 'upper center', frameon = False, prop = font_legend)

plt.legend(title = '(b)', frameon = False, title_fontsize = legendsize)
plt.text(2, 1700, r'Outside Cavity', fontsize = 32)

plt.savefig("figure_4_state.pdf", bbox_inches='tight')