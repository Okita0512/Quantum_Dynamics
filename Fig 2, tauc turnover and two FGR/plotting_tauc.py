import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import voigt_profile
from scipy import integrate
# fig, ax = plt.subplots()
from matplotlib.pyplot import MultipleLocator, tick_params
import matplotlib.ticker as ticker
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

# ================= global ====================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# ==============================================================================================

# linewidth and lineshape
lw = 5.0
lsp = 'o-'
# size for legend
legendsize = 48        
font_legend = {'family':'roman', 'weight': 'roman', 'size': 19}
# axis label size
lsize = 30             
txtsize = 24
# tick length
lmajortick = 15
lminortick = 5
# shared y_lim
y1, y2 = 0.95, 1.45
y_major_locator = MultipleLocator(0.1)
y_minor_locator = MultipleLocator(0.05)

unitlen = 7
fig = plt.figure(figsize=(3 * unitlen, 1 * unitlen), dpi = 128)
fig.subplots_adjust(wspace = 0.0)

# ==============================================================================================

etac = 0.05                                 # default light-matter coupling strength
Rij_2 = 0.214                               # transition dipole moment
wc = 1172.2 * cm_to_au                      # default cavity frequency
w0 = 1172.2 * cm_to_au                      # system vibration energy
beta = 1052.6                               # temperature T = 300 K
tau = 1000 * fs_to_au                       # default cavity lifetime
cfactor = 0.5                                  # correction factor
k0_HEOM = 1.267242438468303195e-07

def coth(x):                                # mathematical function, cot(x)
    return 1 / np.tanh(x)

def bose(x):                                # Bose-Einstein distribution
    return cfactor * 1 / (np.exp(beta * x) - 1)

# def sigmoid(x):                             # sigmoid function
#     return 1.0 / (1.0 + np.exp(-x))

def gauss(x, x0, sigma_2):                  # gaussian distribution, with center x0 and variance sigma_2

    # Lorentz lineshape
    gamma = np.sqrt(sigma_2)
    return (1 / np.pi) * gamma / ((x0 - x)**2 + gamma**2)

    # Gaussian lineshape
#    return (1 / np.sqrt(2 * np.pi * sigma_2)) * np.exp(- (x - x0)**2 / (2 * sigma_2))

    # Gaussian-Lorentzian
#    m = 0.3
#    return m * (1 / np.sqrt(2 * np.pi * sigma_2)) * np.exp(- (x - x0)**2 / (2 * sigma_2)) + (1 - m) * (1 / np.pi) * np.sqrt(sigma_2) / ((x0 - x)**2 + sigma_2)

    # Voigt lineshape
#    return voigt_profile(x - x0, np.sqrt(sigma_2), np.sqrt(sigma_2))

def gen_jw(w, omega_c, eta_c, tau):  # the effective spectral density function
    Gamma = 1. / tau
    return (2 * eta_c**2 * omega_c**3 * Gamma * w) / ((omega_c**2 - w**2)**2 + (Gamma * w)**2)

def Drude(x):                               # the molecular bath spectral density function, J_v(w)
    lam = 83.7 * cm_to_au
    gam = 200 * cm_to_au
    return (2 * lam * gam * x / (x**2 + gam**2)) * coth(beta * x / 2)

"""
sigma^2 = (1 / pi) \int_0^{\infty} dw J_v (w) coth(beta w / 2)
"""

# to get the variance
Rij = 0.231
wi = np.linspace(1e-10, 200 * cm_to_au, 1000000)     # for intergration. Better to be larger (at least 10^3)
y = Drude(wi)
sigma_2 = integrate.trapz(y, wi)
sigma_2 = Rij**2 * sigma_2 / (np.pi)
print("sigma_2 = ", np.sqrt(sigma_2) / cm_to_au, '\t cm^-1')

# ==============================================================================================
#                                      Fig 1a: lossy limit    
# ==============================================================================================

plt.subplot(1,3,1)

wc_scan = np.linspace(600 * cm_to_au, 1800 * cm_to_au, 100)     # data points
wc_scan_2 = np.linspace(0.00001 * wc, 100 * wc, 100000)        # for intergration, Better to be larger (at least 10^5)
# prepare the FGR_lossy data for plot
tau = 100 * fs_to_au
rate_6 = []
for z in wc_scan:
    def intergrant(x):
        return 2 * Rij_2**2 * gen_jw(x, z, etac, tau) * gauss(x, w0, sigma_2) * bose(w0)
    y = intergrant(wc_scan_2)
    rate_w = integrate.trapz(y, wc_scan_2)
    rate_6.append(1 + rate_w / k0_HEOM)

tau = 50 * fs_to_au
rate_7 = []
for z in wc_scan:
    def intergrant(x):
        return 2 * Rij_2**2 * gen_jw(x, z, etac, tau) * gauss(x, w0, sigma_2) * bose(w0)
    y = intergrant(wc_scan_2)
    rate_w = integrate.trapz(y, wc_scan_2)
    rate_7.append(1 + rate_w / k0_HEOM)

tau = 20 * fs_to_au
rate_8 = []
for z in wc_scan:
    def intergrant(x):
        return 2 * Rij_2**2 * gen_jw(x, z, etac, tau) * gauss(x, w0, sigma_2) * bose(w0)
    y = intergrant(wc_scan_2)
    rate_w = integrate.trapz(y, wc_scan_2)
    rate_8.append(1 + rate_w / k0_HEOM)

tau = 10 * fs_to_au
rate_9 = []
for z in wc_scan:
    def intergrant(x):
        return 2 * Rij_2**2 * gen_jw(x, z, etac, tau) * gauss(x, w0, sigma_2) * bose(w0)
    y = intergrant(wc_scan_2)
    rate_w = integrate.trapz(y, wc_scan_2)
    rate_9.append(1 + rate_w / k0_HEOM)

color_a1 = "darkred"
color_a2 = "indianred"
color_a3 = "orangered"
color_a4 = "darkorange"

# prepare the HEOM data and plot
resonance_date = []

data = np.loadtxt('PSD_es=0.1_ec=0.05_tauc=1.txt', dtype=float)
resonance_date.append(data[10, 1])

data = np.loadtxt('PSD_es=0.1_ec=0.05_tauc=10.txt', dtype=float)
plt.plot(data[:, 0], data[:, 1] / k0_HEOM, lsp, markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color_a1)
resonance_date.append(data[10, 1])

data = np.loadtxt('PSD_es=0.1_ec=0.05_tauc=20.txt', dtype=float)
plt.plot(data[:, 0], data[:, 1] / k0_HEOM, lsp, markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color_a2)
resonance_date.append(data[10, 1])

data = np.loadtxt('PSD_es=0.1_ec=0.05_tauc=50.txt', dtype=float)
plt.plot(data[:, 0], data[:, 1] / k0_HEOM, lsp, markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color_a3)
resonance_date.append(data[10, 1])

data = np.loadtxt('PSD_es=0.1_ec=0.05_tauc=100.txt', dtype=float)
plt.plot(data[:, 0], data[:, 1] / k0_HEOM, lsp, markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color_a4)
resonance_date.append(data[10, 1])

data = np.loadtxt('PSD_es=0.1_ec=0.05_tauc=200.txt', dtype=float)
resonance_date.append(data[10, 1])

data = np.loadtxt('PSD_es=0.1_ec=0.05_tauc=500.txt', dtype=float)
resonance_date.append(data[10, 1])

data = np.loadtxt('PSD_es=0.1_ec=0.05_tauc=1000.txt', dtype=float)
resonance_date.append(data[10, 1])

data = np.loadtxt('PSD_es=0.1_ec=0.05_tauc=2000.txt', dtype=float)
resonance_date.append(data[10, 1])

data = np.loadtxt('PSD_es=0.1_ec=0.05_tauc=10000.txt', dtype=float)
resonance_date.append(data[10, 1])

data = np.loadtxt('Discrete_es=0.1_ec=0.05_tauc=inf.txt', dtype=float)
resonance_date.append(data[10, 1])

# plot FGR rate
x_axis = wc_scan / cm_to_au
plt.plot(x_axis, [1.0] * len(x_axis), "--", linewidth = lw, color = 'black', label = r"Outside Cavity")
plt.plot(x_axis, rate_9, linewidth = lw, color = color_a1, label = r"$\tau_\mathrm{c} = 10\ \mathrm{fs}$")
plt.plot(x_axis, rate_8, linewidth = lw, color = color_a2, label = r"$20\ \mathrm{fs}$")
plt.plot(x_axis, rate_7, linewidth = lw, color = color_a3, label = r"$50\ \mathrm{fs}$")
plt.plot(x_axis, rate_6, linewidth = lw, color = color_a4, label = r"$100\ \mathrm{fs}$")

# scale for major and minor locator
x_major_locator = MultipleLocator(200)
x_minor_locator = MultipleLocator(50)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = lmajortick, labelsize = 10, pad = 10)
ax.tick_params(which = 'minor', length = lminortick)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

x1, x2 = 660, 1740

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
ax.legend(frameon = False, loc = 'upper left', prop = font_legend, markerscale = 1)
# plt.text(750, 1.21, r'$\times$ %s' %cfactor, color = 'black', size = 24)
plt.text(1300, 1.3, r'$\tau_\mathrm{c} \ll \Omega_\mathrm{R}^{-1}$', color = 'black', size = 32)
plt.legend(title = '(a) ', frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                      Fig 1b: tau_c scan    
# ==============================================================================================

plt.subplot(1,3,2)

rate_L = []
rate_R = []
rate_C = []
tauc_data1 = [0.5, 1, 2, 5, 10, 20, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 170, 190, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1200, 1500, 2000, 3000, 4000, 5000, 10000, 20000, 50000, 100000]
for z in tauc_data1:
    def intergrant(x):
        Rij_2 = 0.214
        return 2 * Rij_2**2 * gen_jw(x, wc, etac, z * fs_to_au) * gauss(x, w0, sigma_2) * bose(w0)
    y = intergrant(wc_scan_2)
    rate_wL = integrate.trapz(y, wc_scan_2)
    k1 = bose(wc) / (z * fs_to_au)
    Rabi = 2 * etac * Rij * wc
    k2 = 0.5 * np.pi * Rabi**2 * gauss(wc, w0, sigma_2) * bose(wc)
    rate_wR = k1 * k2 / (k1 + k2)
    rate_L.append(1 + rate_wL / k0_HEOM)
    rate_R.append(1 + rate_wR / k0_HEOM)
    rate_wC = rate_wL * rate_wR / (rate_wL + rate_wR)
    rate_C.append(1 + rate_wC / k0_HEOM)

# plot HEOM data
tauc_data2 = [1, 10, 20, 50, 100, 200, 500, 1000, 2000, 10000, 100000]
data = np.zeros((len(tauc_data2), 2), dtype = float)
data[:, 0] = tauc_data2
data[:, 1] = resonance_date

plt.semilogx(data[:, 0], data[:, 1] / k0_HEOM, 'o', markersize = 15, markerfacecolor = 'white', color = 'navy', label = "HEOM")
plt.semilogx(tauc_data1, rate_L, "-", linewidth = lw, color = 'red', label = r"$k_\mathrm{VSC}$")
plt.semilogx(tauc_data1, rate_R, "-", linewidth = lw, color = 'blue', label = r"$\tilde{k}_\mathrm{VSC}$")
plt.semilogx(tauc_data1, rate_C, "--", linewidth = lw, color = 'gold', label = r"$k^\mathrm{int}_\mathrm{VSC}$")

# plot lossy and lossless marks
k_min = 1.0
k_max = 1.0
tau_0 = np.linspace(5 * 1e-1, 30, 10000)
plt.plot(tau_0, [k_min] * len(tau_0), '--', linewidth = lw, color = 'red')
tau_1 = np.linspace(1e3, 1e5, 10000)
plt.plot(tau_1, [k_max] * len(tau_1), '--', linewidth = lw, color = 'blue')
plt.text(1.2, 0.965, r'lossy limit', color = 'red', size = 24)
plt.text(1000, 0.965, r'lossless limit', color = 'blue', size = 24)

ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = lmajortick, labelsize = 10, pad = 10)
ax.tick_params(which = 'minor', length = lminortick)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = lsize)
plt.tick_params(axis = 'y', which = 'both', direction = 'in', labelsize = 0)
plt.xlim(5 * 1e-1, 1e5)
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

ax.set_xlabel(r'$\tau_\mathrm{c}~ (\mathrm{fs})$', size = txtsize)
ax.legend(frameon = False, loc = 'upper left', prop = font_legend, markerscale = 1)
# plt.text(1.5, 1.23, r'$\times$ %s' %cfactor, color = 'black', size = 24)
plt.legend(title = '(b) ', loc = 'upper right', frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                      Fig 1c: lossless limit    
# ==============================================================================================

plt.subplot(1,3,3)

tau = 500 * fs_to_au
rate_5 = []
for z in wc_scan:
    nwc = bose(z)
#    nwc = bose(w0)
    k1 = nwc / tau
    Rabi = 2 * etac * Rij * z
    k2 = 0.5 * np.pi * Rabi**2 * gauss(z, w0, sigma_2) * nwc
    rate_w = k1 * k2 / (k1 + k2)
    rate_5.append(1 + rate_w / k0_HEOM)

tau = 1000 * fs_to_au
rate_6 = []
for z in wc_scan:
    nwc = bose(z)
#    nwc = bose(w0)
    k1 = nwc / tau
    Rabi = 2 * etac * Rij * z
    k2 = 0.5 * np.pi * Rabi**2 * gauss(z, w0, sigma_2) * nwc
    rate_w = k1 * k2 / (k1 + k2)
    rate_6.append(1 + rate_w / k0_HEOM)

tau = 2000 * fs_to_au
rate_7 = []
for z in wc_scan:
    nwc = bose(z)
#    nwc = bose(w0)
    k1 = nwc / tau
    Rabi = 2 * etac * Rij * z
    k2 = 0.5 * np.pi * Rabi**2 * gauss(z, w0, sigma_2) * nwc
    rate_w = k1 * k2 / (k1 + k2)
    rate_7.append(1 + rate_w / k0_HEOM)

tau = 10000 * fs_to_au
rate_8 = []
for z in wc_scan:
    nwc = bose(z)
#    nwc = bose(w0)
    k1 = nwc/ tau
    Rabi = 2 * etac * Rij * z
    k2 = 0.5 * np.pi * Rabi**2 * gauss(z, w0, sigma_2) * nwc
    rate_w = k1 * k2 / (k1 + k2)
    rate_8.append(1 + rate_w / k0_HEOM)

tau = 1e10 * fs_to_au
rate_9 = []
for z in wc_scan:
#    nwc = bose(z)
    nwc = bose(w0)
    k1 = nwc / tau
    Rabi = 2 * etac * Rij * z
    k2 = 0.5 * np.pi * Rabi**2 * gauss(z, w0, sigma_2) * nwc
    rate_w = k1 * k2 / (k1 + k2)
    rate_9.append(1 + rate_w / k0_HEOM)

color_b1 = "cyan"
color_b2 = "dodgerblue"
color_b3 = "b"
color_b4 = "mediumpurple"
color_b5 = "darkviolet"

# prepare the HEOM data and plot
data = np.loadtxt('PSD_es=0.1_ec=0.05_tauc=500.txt', dtype=float)
plt.plot(data[:, 0], data[:, 1] / k0_HEOM, lsp, markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color_b5)

data = np.loadtxt('PSD_es=0.1_ec=0.05_tauc=1000.txt', dtype=float)
plt.plot(data[:, 0], data[:, 1] / k0_HEOM, lsp, markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color_b4)

data = np.loadtxt('PSD_es=0.1_ec=0.05_tauc=2000.txt', dtype=float)
plt.plot(data[:, 0], data[:, 1] / k0_HEOM, lsp, markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color_b3)

data = np.loadtxt('PSD_es=0.1_ec=0.05_tauc=10000.txt', dtype=float)
plt.plot(data[:, 0], data[:, 1] / k0_HEOM, lsp, markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color_b2)

data = np.loadtxt('Discrete_es=0.1_ec=0.05_tauc=inf.txt', dtype=float)
plt.plot(data[:, 0], data[:, 1] / k0_HEOM, lsp, markersize = 10, linewidth = 1, markerfacecolor = 'white', color = color_b1)

# plot FGR rate
x_axis = wc_scan / cm_to_au
plt.plot(x_axis, rate_9, linewidth = lw, color = color_b1, label = r"$\tau_\mathrm{c} = \infty$")
plt.plot(x_axis, rate_8, linewidth = lw, color = color_b2, label = r"$10000\ \mathrm{fs}$")
plt.plot(x_axis, rate_7, linewidth = lw, color = color_b3, label = r"$2000\ \mathrm{fs}$")
plt.plot(x_axis, rate_6, linewidth = lw, color = color_b4, label = r"$1000\ \mathrm{fs}$")
plt.plot(x_axis, rate_5, linewidth = lw, color = color_b5, label = r"$500\ \mathrm{fs}$")

# x and y range of plotting 
x1, x2 = 660, 1740

# scale for major and minor locator
x_major_locator = MultipleLocator(200)
x_minor_locator = MultipleLocator(50)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = lmajortick, labelsize = 10, pad = 10)
ax.tick_params(which = 'minor', length = lminortick)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = lsize)
plt.tick_params(axis = 'y', which = 'both', direction = 'in', labelsize = 0)
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
# ax.set_ylabel(r'$k / k_0$', size = txtsize)
ax.legend(frameon = False, loc = 'upper left', prop = font_legend, markerscale = 1)
# plt.text(750, 1.21, r'$\times$ %s' %cfactor, color = 'black', size = 24)
plt.text(1300, 1.3, r'$\tau_\mathrm{c} \gg \Omega_\mathrm{R}^{-1}$', color = 'black', size = 32)
plt.legend(title = '(c) ', frameon = False, title_fontsize = legendsize)



plt.savefig("Fig_Lorentzian.pdf", bbox_inches='tight')

print("done")