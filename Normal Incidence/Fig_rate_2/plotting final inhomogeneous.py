import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
from scipy import integrate
from matplotlib.pyplot import MultipleLocator, tick_params
# fig = plt.figure(figsize=(10,5),dpi=80)
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

lw = 3.0
axis_size = 28
unitlen = 7
legendsize = 48         # size for legend
font_legend = {'family':'Times New Roman',
        'style':'normal', # 'italic'
        'weight':'normal', #or 'blod'
        'size':18
        }

# axis label size
lsize = 30             
txtsize = 36
# tick length
lmajortick = 15
lminortick = 5

fig, ax = plt.subplots(figsize=(2.0 * unitlen, 1.0 * unitlen), dpi = 512, sharey = 'row')
fig.subplots_adjust(wspace = 0.0) # hspace = 0.25, 

# ==============================================================================================
#                                       Global Parameters     
# ==============================================================================================
conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# ==============================================================================================
#                                       Plotting Fig 6a     
# ==============================================================================================

plt.subplot(1,2,1)

wc = 1190 * cm_to_au                      # default cavity frequency
w0 = 1190 * cm_to_au                      # system vibration energy
Eb = 2250 * cm_to_au
beta = 1052.6                               # temperature T = 300 K
tau_c = 200 * fs_to_au                       # cavity lifetime
gamma_c = 1e10 * cm_to_au                   # loss bath cutoff frequency

n_0 = 1 / (np.exp(beta * w0) - 1) * 0.4
print('n_0(w0) = ', n_0)

k_0 = 5.946954192406803128e-08

def coth(x):                                # mathematical function, cot(x)
    return 1 / np.tanh(x)

def gauss(x, x0, sigma_2):                  # gaussian distribution, with center x0 and variance sigma_2
    return (1 / np.sqrt(2 * np.pi * sigma_2)) * np.exp(- (x - x0)**2 / (2 * sigma_2))

def λc(tau_c, ωc, gamma):                     # function to calculate loss bath reorganization energy from cavity lifetime
    return (1.0 - np.exp(- beta * ωc)) * (gamma**2 + ωc**2) / (2 * tau_c * gamma)

print('effective cavity lifetime', gamma_c / (2 * λc(tau_c, wc, gamma_c)) / fs_to_au, 'fs')

def gen_jw(w, omega_c, eta_c, lam, gamma):  # the effective spectral density function

    J0 = (2 * lam * gamma * w / (w**2 + gamma**2)) # * (2 * omega_c)     # secondary bath dissipation operator: np.sqrt(2 w_c) q_c, which is just (a + a^+)
    zeta = np.sqrt(2 / omega_c) * eta_c

    return ((omega_c**4 * zeta**2 * J0) / ((omega_c**2 - w**2 + (w * J0 / gamma))**2 + (J0)**2))

def Drude(x):                               # the molecular bath spectral density function, J_v(w)
    lam = 83.7 * cm_to_au / 1836
    gam = 200 * cm_to_au
    return (2 * lam * gam * x / (x**2 + gam**2)) * coth(beta * x / 2)

"""
sigma^2 = (1 / pi) \int_0^{\infty} dw J_v (w) coth(beta w / 2)
"""

# to get the variance
Rij = 9.87
wi = np.linspace(1e-10, 200 * cm_to_au, 10000000)     # for intergration. Better to be larger (at least 10^3)
y = Drude(wi)
sigma_2 = integrate.trapz(y, wi)
# sigma_2 = (0.01 * cm_to_au)**2 # 
sigma_2 = Rij**2 * sigma_2 / (np.pi)
print("sigma_2 = ", np.sqrt(sigma_2) / cm_to_au, '\t cm^-1')

# evaluating inhomogeneous broadening
"""
k_FGR(w') = \int_0^{\infty} dw kappa(w, w') * G(w - w0),   Note: w' is the cavity frequency and need to scan from 600 cm^-1 to 1800 cm^-1

where 

kappa(w, w') = 2 * mu_0**2 * J_eff(w, w') * n_0
G(w - w0) is the Gaussian distribution centered at w0, with variance sigma_2

"""

mu_0 = 9.14

etac = 0.0003125                                # light-matter coupling strength
rate_1 = []
wc_scan = np.array([600, 650, 700, 750, 800, 850, 900, 950, 1000, 
           1020, 1040, 1060, 1080, 1100, 1110, 1120, 1130, 1140, 1150, 
           1155, 1160, 1165, 1170, 1175, 1180, 1185, 1190, 1195, 1200,
           1210, 1220, 1230, 1240, 1260, 1280, 1300, 1325, 1350, 1400, 
           1500, 1600, 1700, 1800]) * cm_to_au
# wc_scan_2 = np.linspace(0.00001 * wc, 100 * wc, 1000000)        # for intergration, Better to be larger (at least 10^5)
wc_scan_2 = np.linspace(0.8 * wc, 1.2 * wc, 100)        # for intergration, Better to be larger (at least 10^5)
for z in wc_scan:
    lambda_c = λc(tau_c, z, gamma_c)          # loss bath reorganization energy
    def intergrant(x):
        return 2 * mu_0**2 * gen_jw(x, z, etac, lambda_c, gamma_c) * gauss(x, w0, sigma_2) * n_0
    y = intergrant(wc_scan_2)
    rate_w = integrate.trapz(y, wc_scan_2)
    rate_1.append(1 + rate_w / k_0)

etac = 0.000625                                # light-matter coupling strength
rate_2 = []
for z in wc_scan:
    lambda_c = λc(tau_c, z, gamma_c)          # loss bath reorganization energy
    def intergrant(x):
        return 2 * mu_0**2 * gen_jw(x, z, etac, lambda_c, gamma_c) * gauss(x, w0, sigma_2) * n_0
    y = intergrant(wc_scan_2)
    rate_w = integrate.trapz(y, wc_scan_2)
    rate_2.append(1 + rate_w / k_0)

etac = 0.0009375                                # light-matter coupling strength
rate_3 = []
for z in wc_scan:
    lambda_c = λc(tau_c, z, gamma_c)          # loss bath reorganization energy
    def intergrant(x):
        return 2 * mu_0**2 * gen_jw(x, z, etac, lambda_c, gamma_c) * gauss(x, w0, sigma_2) * n_0
    y = intergrant(wc_scan_2)
    rate_w = integrate.trapz(y, wc_scan_2)
    rate_3.append(1 + rate_w / k_0)

etac = 0.00125                                # light-matter coupling strength
rate_4 = []
for z in wc_scan:
    lambda_c = λc(tau_c, z, gamma_c)          # loss bath reorganization energy
    def intergrant(x):
        return 2 * mu_0**2 * gen_jw(x, z, etac, lambda_c, gamma_c) * gauss(x, w0, sigma_2) * n_0
    y = intergrant(wc_scan_2)
    rate_w = integrate.trapz(y, wc_scan_2)
    rate_4.append(1 + rate_w / k_0)

x_axis = wc_scan / cm_to_au
plt.plot(x_axis, [1.0] * len(x_axis), linewidth = lw, color = 'black', label = r"outside cavity")
plt.plot(x_axis, rate_1, linewidth = lw, color = 'blue', label = r"$\eta_\mathrm{c} = 3.125 \times 10^{-4}$")
plt.plot(x_axis, rate_2, linewidth = lw, color = 'cyan', label = r"$6.25 \times 10^{-4}$")
plt.plot(x_axis, rate_3, linewidth = lw, color = 'green', label = r"$9.375 \times 10^{-4}$")
plt.plot(x_axis, rate_4, linewidth = lw, color = 'red', label = r"$1.25 \times 10^{-3}$")

plt.text(1360, 1.9, r'$\omega_\mathrm{0}$', color = 'black', size = 20)
plt.text(1270, 1.8, r'$1190\ \mathrm{cm}^{-1}$', color = 'black', size = 20)

# x and y range of plotting 
x1, x2 = 600, 1740
y1, y2 = 0.95, 2.2     # y-axis range: (y1, y2)

# scale for major and minor locator
x_major_locator = MultipleLocator(300)
x_minor_locator = MultipleLocator(60)
y_major_locator = MultipleLocator(0.5)
y_minor_locator = MultipleLocator(0.1)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = lmajortick, labelsize = 10, pad = 10)
ax.tick_params(which = 'minor', length = lminortick)

ax.vlines([1190], 0.95, 2.2, linestyles = 'dashed', colors = ['black'], lw = 3.0) 

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(labelsize = lsize, which = 'both', direction = 'in')
plt.xlim(x1, x2)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = lmajortick)
ax2.tick_params(which = 'minor', length = lminortick)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

ax.set_xlabel(r'$\omega_\mathrm{c}~ (\mathrm{cm}^{-1})$', size = txtsize)
ax.set_ylabel(r'$k / k_0$', size = txtsize)
# ax.legend(frameon = False, loc = 'upper left', prop = font_legend, markerscale = 1)
plt.legend(title = '(a)', frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                       Plotting Fig 6b     
# ==============================================================================================

plt.subplot(1,2,2)

# ===== Auxiliary functions =====
def cot(x):
    return np.cos(x) / np.sin(x)

def omega_k(theta, wc):
    return wc * np.sqrt(1 + np.tan(theta)**2)

def brownian(w, alpha, wcav):  # the effective spectral density function
    return ((2 * wcav**2 * alpha * w) / ((wcav**2 - w**2)**2 + (alpha * w)**2))

Gamma_perpendicular = 1.0 / tau_c
tau_0 = 3.33333333333e1 * fs_to_au
temp = 1. / beta

w_scan = np.linspace(x1 * cm_to_au, x2 * cm_to_au, 1000)    # scan the w0 value
# wc_scan_2 = np.linspace(0.8 * wc, 1.2 * wc, 100)
dw2 = wc_scan_2[1] - wc_scan_2[0]

Ngrids = int(4e4)
# Ngrids = int(4e6)
rescaling = 0.4

plt.plot(x_axis, [1.0] * len(x_axis), linewidth = lw, color = 'black', label = r"outside cavity")

spd = []
etac = 0.0003125

for z in w_scan:
    def DOS(tau_c_0, w):
        Gamma_parallel = (np.sqrt(np.abs(w**2 - z**2)) / w) / tau_c_0
        return w * Gamma_perpendicular * np.exp(- w / temp) / (Gamma_perpendicular + Gamma_parallel)
    w_int = np.logspace(np.log(z), np.log(5 * z), Ngrids, base = np.e)
    def lam_mu(etac):
        lam = np.sqrt(2 * z) * etac
        lamc_mu_0 = mu_0 * lam
        return lamc_mu_0
    lamc_mu_0 = lam_mu(etac)
    def integrant_2(w):
        jw_2 = DOS(tau_0, w)
        return jw_2
    y_2 = integrant_2(w_int)
    norm = integrate.trapz(y_2, w_int)
#    print(norm)
    Jz = 0
    for w_0 in wc_scan_2:
        def integrant(w):
            jw = brownian(w_0, Gamma_perpendicular, w) * DOS(tau_0, w) * np.exp(- w_0 / temp) * rescaling
            return jw
        y = integrant(w_int) * gauss(w_0, w0, sigma_2) * dw2
        Jz += integrate.trapz(y, w_int)
    spd.append(1 + (Jz * lamc_mu_0**2 / norm) / k_0)

plt.plot(w_scan / cm_to_au, spd, '-', linewidth = lw, color = 'blue', label = r'$\eta_\mathrm{c} = 3.125 \times 10^{-4}$')

spd = []
etac = 0.000625

for z in w_scan:
    def DOS(tau_0, w):
        Gamma_parallel = (np.sqrt(np.abs(w**2 - z**2)) / w) / tau_0
        return w * Gamma_perpendicular * np.exp(- w / temp) / (Gamma_perpendicular + Gamma_parallel)
    w_int = np.logspace(np.log(z), np.log(5 * z), Ngrids, base = np.e)
    def lam_mu(etac):
        lam = np.sqrt(2 * z) * etac
        lamc_mu_0 = mu_0 * lam
        return lamc_mu_0
    lamc_mu_0 = lam_mu(etac)
    def integrant_2(w):
        jw_2 = DOS(tau_0, w)
        return jw_2
    y_2 = integrant_2(w_int)
    norm = integrate.trapz(y_2, w_int)
    Jz = 0
    for w_0 in wc_scan_2:
        def integrant(w):
            jw = brownian(w_0, Gamma_perpendicular, w) * DOS(tau_0, w) * np.exp(- w_0 / temp) * rescaling
            return jw
        y = integrant(w_int) * gauss(w_0, w0, sigma_2) * dw2
        Jz += integrate.trapz(y, w_int)
    spd.append(1 + (Jz * lamc_mu_0**2 / norm) / k_0)

plt.plot(w_scan / cm_to_au, spd, '-', linewidth = lw, color = 'cyan', label = r'$6.25 \times 10^{-4}$')

spd = []
etac = 0.0009375

for z in w_scan:
    def DOS(tau_0, w):
        Gamma_parallel = (np.sqrt(np.abs(w**2 - z**2)) / w) / tau_0
        return w * Gamma_perpendicular * np.exp(- w / temp) / (Gamma_perpendicular + Gamma_parallel)
    w_int = np.logspace(np.log(z), np.log(5 * z), Ngrids, base = np.e)
    def lam_mu(etac):
        lam = np.sqrt(2 * z) * etac
        lamc_mu_0 = mu_0 * lam
        return lamc_mu_0
    lamc_mu_0 = lam_mu(etac)
    def integrant_2(w):
        jw_2 = DOS(tau_0, w)
        return jw_2
    y_2 = integrant_2(w_int)
    norm = integrate.trapz(y_2, w_int)
    Jz = 0
    for w_0 in wc_scan_2:
        def integrant(w):
            jw = brownian(w_0, Gamma_perpendicular, w) * DOS(tau_0, w) * np.exp(- w_0 / temp) * rescaling
            return jw
        y = integrant(w_int) * gauss(w_0, w0, sigma_2) * dw2
        Jz += integrate.trapz(y, w_int)
    spd.append(1 + (Jz * lamc_mu_0**2 / norm) / k_0)

plt.plot(w_scan / cm_to_au, spd, '-', linewidth = lw, color = 'green', label = r'$9.375 \times 10^{-4}$')

spd = []
etac = 0.00125

for z in w_scan:
    def DOS(tau_0, w):
        Gamma_parallel = (np.sqrt(np.abs(w**2 - z**2)) / w) / tau_0
        return w * Gamma_perpendicular * np.exp(- w / temp) / (Gamma_perpendicular + Gamma_parallel)
    w_int = np.logspace(np.log(z), np.log(5 * z), Ngrids, base = np.e)
    def lam_mu(etac):
        lam = np.sqrt(2 * z) * etac
        lamc_mu_0 = mu_0 * lam
        return lamc_mu_0
    lamc_mu_0 = lam_mu(etac)
    def integrant_2(w):
        jw_2 = DOS(tau_0, w)
        return jw_2
    y_2 = integrant_2(w_int)
    norm = integrate.trapz(y_2, w_int)
    Jz = 0
    for w_0 in wc_scan_2:
        def integrant(w):
            jw = brownian(w_0, Gamma_perpendicular, w) * DOS(tau_0, w) * np.exp(- w_0 / temp) * rescaling
            return jw
        y = integrant(w_int) * gauss(w_0, w0, sigma_2) * dw2
        Jz += integrate.trapz(y, w_int)
    spd.append(1 + (Jz * lamc_mu_0**2 / norm) / k_0)

plt.plot(w_scan / cm_to_au, spd, '-', linewidth = lw, color = 'red', label = r'$1.25 \times 10^{-3}$')

plt.text(750, 1.625, r'$\times$0.4', color = 'goldenrod', size = 24)
plt.text(1360, 1.9, r'$\omega_\mathrm{0}$', color = 'black', size = 20)
plt.text(1270, 1.8, r'$1190\ \mathrm{cm}^{-1}$', color = 'black', size = 20)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 10, pad = 10)
ax.tick_params(which = 'minor', length = 5)

ax.vlines([1190], 0.95, 1.9, linestyles = 'dashed', colors = ['black'], lw = 3.0) 

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params('x', labelsize = 30, which = 'both', direction = 'in')
plt.tick_params('y', labelsize = 0, which = 'both', direction = 'in')
plt.xlim(x1, x2)
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
ax.set_xlabel(r'$\omega_\mathrm{c}$ ($\mathrm{cm}^{-1}$)', size = txtsize)
# ax.set_ylabel(r'$k/k_0$', size = 36)
ax.legend(loc = 'upper left', frameon = False, prop = font_legend)
plt.legend(title = '(b)', frameon = False, title_fontsize = legendsize)









plt.savefig("figure_rate_2_ihm.pdf", bbox_inches='tight')