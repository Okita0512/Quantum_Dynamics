import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
from matplotlib.pyplot import MultipleLocator, tick_params
from plt_arrowline import * #我测试了下，采用这种方式导入，能自动导入源文件依赖的库函数
fig = plt.figure(figsize=(10,5),dpi=80)
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

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
L = 100.0
R = np.linspace(- L, L, Ngrid)
dx = R[1] - R[0]

# Model parameters: DW2
m_s = 1
omega_b = 1000 * cm_to_au
E_b = 2250 * cm_to_au

# ===== Cavity parameters =====
omega_c = 1190 * cm_to_au         # cavity frequency. Note that in this model, the energy gap is around 1140 cm^-1
lamc = 100 * cm_to_au
eta_c = lamc / np.sqrt(2 * omega_c)                     # light-matter-coupling strength. Set as 0 when cavity is turned off

# ===== Drude-Lorentz model =====
temp    = 300 / au_to_K                             # temperature

# Bath I parameters, Drude-Lorentz model
gamma_1   = 200 * cm_to_au                      # bath characteristic frequency
ratio = 0.1                                     # the value of etas / omega_b, tune it from 0.02 to 2.0
lambda_1 = ratio * m_s * omega_b * gamma_1 / 2        # reorganization energy

def λc(tau, ωc, gamma, temp):

    beta = 1. / temp
    lr = 1. / tau     # cavity loss rate, defined as 1 / relaxation time

    return lr * (1.0 - np.exp(- beta * ωc)) * ( gamma**2 + ωc**2 ) / (2 * gamma) # (2 * gamma)

# Bath II parameters, Brownian Oscillator
gamma_2 = 1e6 * cm_to_au                        # bath characteristic frequency  
tau_c = 100 * fs_to_au                           # bath relaxation time
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

    J0 = (2 * lam * gamma * w / (w**2 + gamma**2))
    zeta = np.sqrt(2 / omega_c) * eta_c

    return (omega_c**4 * zeta**2 * J0) / ((omega_c**2 - w**2 + (w * J0 / gamma))**2 + (J0)**2) * np.heaviside(w, 0)

def Drude(w, lam, gam):
    return 2 * lam * gam * w / (w**2 + gam**2) * np.heaviside(w, 0)

# ==============================================================================================
#                                       Plotting Fig 1a     
# ==============================================================================================

lw = 3.0
unitlen = 9
legendsize = 64         # size for legend
font_legend = {'weight': 'roman', 'size': 24}

fig = plt.figure(figsize=(1.0 * unitlen, 0.9 * unitlen), dpi = 512)
# fig, ax = plt.subplots(1, 10, figsize=(1.88 * unitlen, 0.9 * unitlen), dpi = 512, sharey = 'row')
# fig.subplots_adjust(hspace = 0.25) # , wspace = 0.0

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
del1 = (eigenvalue[1] - eigenvalue[0]) / 2
eigenvalue[0] = ave1
eigenvalue[1] = ave1

ave2 = (eigenvalue[2] + eigenvalue[3]) / 2
del2 = (eigenvalue[3] - eigenvalue[2]) / 2
eigenvalue[2] = ave2
eigenvalue[3] = ave2

vL1 = (eigenvec[:, 0] + eigenvec[:, 1]) / np.sqrt(2)
vR1 = (eigenvec[:, 0] - eigenvec[:, 1]) / np.sqrt(2)
eigenvec[:, 0] = vL1
eigenvec[:, 1] = vR1

vL2 = (eigenvec[:, 2] - eigenvec[:, 3]) / np.sqrt(2)
vR2 = (eigenvec[:, 2] + eigenvec[:, 3]) / np.sqrt(2)
eigenvec[:, 2] = vL2
eigenvec[:, 3] = vR2
# =======================================
print((ave2 - ave1) / cm_to_au)
print(del1 / cm_to_au)
print(del2 / cm_to_au)

del temp1
del temp2
del ordered_list

# check the correctness
ratio_1 = cm_to_au
ratio_2 = 0.0003

color1 = 'silver'
color2 = 'gray'
color3 = '#A1A9D0' # purple
color4 = '#8ECFC9' # green
color5 = '#c82423' # blue
color6 = '#2878b5' # red

plt.plot(R, [(eigenvalue[5] - eigenvalue[0]) / ratio_1] * len(R), '-', linewidth = 1.0, color = color1, alpha = 0.5)
plt.plot(R, [(eigenvalue[4] - eigenvalue[0]) / ratio_1] * len(R), '-', linewidth = 1.0, color = color2, alpha = 0.5)
plt.plot(R, [(eigenvalue[3] - eigenvalue[0]) / ratio_1] * len(R), '-', linewidth = 1.0, color = color3)
plt.plot(R, [(eigenvalue[2] - eigenvalue[0]) / ratio_1] * len(R), '-', linewidth = 1.0, color = color4)
plt.plot(R, [(eigenvalue[1] - eigenvalue[0]) / ratio_1] * len(R), '-', linewidth = 1.0, color = color5)
plt.plot(R, [(eigenvalue[0] - eigenvalue[0]) / ratio_1] * len(R), '-', linewidth = 1.0, color = color6)

plt.plot(R, (V(R) - eigenvalue[0]) / ratio_1, linewidth = lw, color = 'black')

plt.plot(R, (eigenvalue[2] - eigenvalue[0]) / ratio_1 + eigenvec[:, 2] / ratio_2, linewidth = lw, color = color4, label = r"$\|\nu'_\mathrm{L}\rangle$")
plt.plot(R, (eigenvalue[1] - eigenvalue[0]) / ratio_1 + eigenvec[:, 1] / ratio_2, linewidth = lw, color = color5, label = r'$\|\nu_\mathrm{L}\rangle$')
plt.plot(R, (eigenvalue[3] - eigenvalue[0]) / ratio_1 + eigenvec[:, 3] / ratio_2, linewidth = lw, color = color3, label = r"$\|\nu'_\mathrm{R}\rangle$")
plt.plot(R, (eigenvalue[0] - eigenvalue[0]) / ratio_1 + eigenvec[:, 0] / ratio_2, linewidth = lw, color = color6, label = r'$\|\nu_\mathrm{R}\rangle$')

# ratio_new = 100
# plt.plot(R, (eigenvalue[4] - eigenvalue[0]) / ratio_1 + eigenvec[:, 4] / ratio_new, linewidth = lw, color = color2, label = r'$\|\nu_4\rangle$')
# plt.plot(R, (eigenvalue[5] - eigenvalue[0]) / ratio_1 + eigenvec[:, 5] / ratio_new, linewidth = lw, color = color1, label = r'$\|\nu_5\rangle$')

# plt.fill_between(R,(eigenvalue[5] - eigenvalue[0]) / ratio_1, (eigenvalue[5] - eigenvalue[0]) / ratio_1 + eigenvec[:, 5] / ratio_2, color = color1, alpha = .6)
# plt.fill_between(R,(eigenvalue[4] - eigenvalue[0]) / ratio_1, (eigenvalue[4] - eigenvalue[0]) / ratio_1 + eigenvec[:, 4] / ratio_2, color = color2, alpha = .6)
plt.fill_between(R,(eigenvalue[3] - eigenvalue[0]) / ratio_1, (eigenvalue[3] - eigenvalue[0]) / ratio_1 + eigenvec[:, 3] / ratio_2, color = color3, alpha = .3)
plt.fill_between(R,(eigenvalue[2] - eigenvalue[0]) / ratio_1, (eigenvalue[2] - eigenvalue[0]) / ratio_1 + eigenvec[:, 2] / ratio_2, color = color4, alpha = .3)
plt.fill_between(R,(eigenvalue[1] - eigenvalue[0]) / ratio_1, (eigenvalue[1] - eigenvalue[0]) / ratio_1 + eigenvec[:, 1] / ratio_2, color = color5, alpha = .3)
plt.fill_between(R,(eigenvalue[0] - eigenvalue[0]) / ratio_1, (eigenvalue[0] - eigenvalue[0]) / ratio_1 + eigenvec[:, 0] / ratio_2, color = color6, alpha = .3)

plt.text(- 20, 1750, r'Tunneling', color = 'black', fontsize = 28)

def arrow_exp(x_0, y_0, y_1, x):
    return (y_0 - y_1) * (x / x_0)**2 + y_1

# scale for major and minor locator
x_major_locator = MultipleLocator(50)
x_minor_locator = MultipleLocator(10)
y_major_locator = MultipleLocator(500)
y_minor_locator = MultipleLocator(100)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 20)
ax.tick_params(which = 'minor', length = 5)

x_0 = -50
y_0 = 1190
y_1 = 1650
ratio_arrow = 0.85
arrowrange = np.linspace(x_0 * ratio_arrow, - x_0 * ratio_arrow, 1000)
arrowcolor = 'darkgreen'
plt.plot(arrowrange, arrow_exp(x_0, y_0, y_1, arrowrange), lw = 2.5, color = arrowcolor)
arrowline(ax, arrowrange, arrow_exp(x_0, y_0, y_1, arrowrange), style='to', arrow_size = 2, arrow_style='above', arrow_angle = 80, color = arrowcolor)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

# arrowline(ax1,x,y,style='to_back',arrow_size=1.5,d_frac=0.5)
# arrowline(ax1,[-10,2],[0,1],style='middle',arrow_size=4,d_frac=0.1)
# arrowline(ax2,y,10*x,style='equal_d',interval=95,arrow_size=2,color='r',arrow_style='above')
# arrowline(ax2,[-10,2],[0,40],style='to_back',arrow_size=3,arrow_style='below',arrow_angle=50)
# arrowline(ax1,[5,-7],[-1,1])
# arrowline(ax2,x-2*y,x*y*x,style='equal_d',interval=115,color='y',arrow_size=1.25)

x0 = 80
y1, y2 = - 800, 2800

plt.tick_params(labelsize = 30, which = 'both', direction = 'in')
plt.xlim(- x0, x0)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel(r'$\mathrm{R\ (a.u.)}$', size = 36)
ax.set_ylabel(r'$\omega = E - E_0\ (\mathrm{cm}^{-1})$', size = 36)
ax.legend(ncol = 2, loc = 'upper center', frameon = False, edgecolor = 'black', prop = font_legend) # , bbox_to_anchor=(0.5, 0.979) fancybox = False, borderpad = 0.7, 
# plt.legend(title = '(a)   ', frameon = False, title_fontsize = legendsize)



plt.savefig("figure_mechanism.pdf", bbox_inches='tight')