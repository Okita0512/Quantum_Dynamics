import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from matplotlib.pyplot import MultipleLocator, tick_params
from plt_arrowline_left import arrowline as arrowleft
from plt_arrowline_right import arrowline as arrowright
fig, ax = plt.subplots()
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

lw = 3.0
legendsize = 48         # size for legend
font_legend = {'family':'Times New Roman', 'weight': 'roman', 'size': 18}
# axis label size
lsize = 30             
txtsize = 32
# tick length
lmajortick = 15
lminortick = 5
legend_x, legend_y = - 0.103, 1.03

unitlen = 7
fig = plt.figure(figsize=(4.0 * unitlen, 1.05 * unitlen), dpi = 128)
# plt.subplots_adjust(wspace = 0.5)

# ==============================================================================================
#                           Fig 2b: compare spectra and rate profile 
# ==============================================================================================

plt.subplot(1,100,(1,26))

y1, y2 = 0.988, 1.24     # y-axis range: (y1, y2)
x_major_locator = MultipleLocator(300)
x_minor_locator = MultipleLocator(60)
y_major_locator = MultipleLocator(0.1)
y_minor_locator = MultipleLocator(0.02)

ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = lmajortick, labelsize = lsize, pad = 10)
ax.tick_params(which = 'minor', length = lminortick)

ax.set_ylabel(r'$k / k_0$', size = txtsize)
ax.vlines([1172], 0.9, 1.3, linestyles = 'dashed', colors = ['black'], lw = 3.0) 
plt.text(1320, 1.21, r'$\omega_\mathrm{0}$', color = 'black', size = 20)
plt.text(1220, 1.185, r'$1172\ \mathrm{cm}^{-1}$', color = 'black', size = 20)

x_00, x_01 = 920, 700
x_10, x_11 = 1440, 1660
y_0 = 1.06
y_1 = 1.06
arrowrange_0 = np.linspace(x_00, x_01, 1000)
arrowrange_1 = np.linspace(x_10, x_11, 1000)
plt.plot(arrowrange_0, [y_0] * len(arrowrange_0), lw = 4, color = 'blue')
plt.plot(arrowrange_1, [y_1] * len(arrowrange_1), lw = 4, color = 'red')
arrowleft(ax, arrowrange_0 * 0.98, [y_0] * len(arrowrange_0), style = 'to', arrow_size = 3, arrow_style = 'full', arrow_angle = 40, color = 'blue')
arrowright(ax, arrowrange_1 * 1.02, [y_1] * len(arrowrange_1), style = 'to', arrow_size = 3, arrow_style = 'full', arrow_angle = 40, color = 'red')

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(labelsize = 30, which = 'both', direction = 'in')
plt.xlim(600, 1800)
plt.ylim(y1, y2)

# plot the VSC rate profile
data1 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=1000.txt", dtype=float)
line_l, = ax.plot(data1[:, 0], data1[:, 1] / data1[0, 1], 'o-', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = 'dodgerblue')
ax.fill_between(data1[:, 0], data1[:, 1] / data1[0, 1],  [1.0] * len(data1[:, 0]), color = 'dodgerblue', alpha = .4)

# ==============================================================================================

wc = 1172 * cm_to_au                      # default cavity frequency
w0 = 1172 * cm_to_au                      # system vibration energy
Eb = 2120 * cm_to_au
beta = 1052.6                               # temperature T = 300 K
tau_c = 1000 * fs_to_au                       # cavity lifetime
gamma_c = 1e12 * cm_to_au                   # loss bath cutoff frequency
k_0 = 1.267242438468303195e-07

def coth(x):                                # mathematical function, cot(x)
    return 1 / np.tanh(x)

def Bose(x):
    return 1 / (np.exp(beta * x) - 1)

def gauss(x, x0, sigma_2):                  # gaussian distribution, with center x0 and variance sigma_2
    return (1./np.pi) * np.sqrt(sigma_2) / ((x - x0)**2 + sigma_2)
#    return (1 / np.sqrt(2 * np.pi * sigma_2)) * np.exp(- (x - x0)**2 / (2 * sigma_2))

def λc(tau_c, ωc, gamma):                     # function to calculate loss bath reorganization energy from cavity lifetime
    return (1.0 - np.exp(- beta * ωc)) * (gamma**2 + ωc**2) / (2 * tau_c * gamma)

print('effective cavity lifetime', gamma_c / (2 * λc(tau_c, wc, gamma_c)) / fs_to_au, 'fs')

def gen_jw(w, omega_c, eta_c, lam, gamma):  # the effective spectral density function

    J0 = (2 * lam * gamma * w / (w**2 + gamma**2)) # * (2 * omega_c)     # secondary bath dissipation operator: np.sqrt(2 w_c) q_c, which is just (a + a^+)
    zeta = np.sqrt(2 / omega_c) * eta_c

    return ((omega_c**4 * zeta**2 * J0) / ((omega_c**2 - w**2 + (w * J0 / gamma))**2 + (J0)**2))

def Drude(x):                               # the molecular bath spectral density function, J_v(w)
    lam = 83.7 * cm_to_au
    gam = 200 * cm_to_au
    return (2 * lam * gam * x / (x**2 + gam**2)) * coth(beta * x / 2)

"""
sigma^2 = (1 / pi) \int_0^{\infty} dw J_v (w) coth(beta w / 2)
"""

# to get the variance
Rij = 0.231
wi = np.linspace(1e-10, 200 * cm_to_au, 10000000)     # for intergration. Better to be larger (at least 10^3)
y = Drude(wi)
sigma_2 = integrate.trapz(y, wi)
sigma_2 = Rij**2 * sigma_2 / (np.pi)
print("sigma_2 = ", np.sqrt(sigma_2) / cm_to_au, '\t cm^-1')

# evaluating inhomogeneous broadening
"""
k_FGR(w') = \int_0^{\infty} dw kappa(w, w') * G(w - w0),   Note: w' is the cavity frequency and need to scan from 600 cm^-1 to 1800 cm^-1

where 

kappa(w, w') = 2 * mu_0**2 * J_eff(w, w') * n_0
G(w - w0) is the Gaussian distribution centered at w0, with variance sigma_2

"""

mu_0 = 0.214

etac = 0.05                                # light-matter coupling strength
rate_1 = []
wc_scan = np.linspace(600 * cm_to_au, 1800 * cm_to_au, 500)     # data points
for z in wc_scan:
    lambda_c = λc(tau_c, z, gamma_c)          # loss bath reorganization energy
    lamc_mu_0 = mu_0 * np.sqrt(2 * z) * etac
    rate_w = np.pi * lamc_mu_0**2 * z * gauss(z, w0, sigma_2) / (1 + np.pi * lamc_mu_0**2 * z * gauss(z, w0, sigma_2) * tau_c) * Bose(z)
    rate_1.append(1 + 0.5 * rate_w / k_0)

line_l2, = ax.plot(wc_scan / cm_to_au, rate_1, linewidth = lw, color = 'blue', label = r"FGR rates")

# ==============================================================================================

# RHS y-axis
y1, y2 = -5.0, 100.0
ax2 = ax.twinx()
y_major_locator = MultipleLocator(20)
y_minor_locator = MultipleLocator(10)
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = lmajortick, labelsize = lsize)
ax2.tick_params(which = 'minor', length = lminortick)

ax2.set_ylabel(r'Intensity (%)', size = txtsize)

y2_label = ax2.get_yticklabels()
[y2_label_temp.set_fontname('Times New Roman') for y2_label_temp in y2_label]

plt.tick_params(labelsize = lsize, which = 'both', direction = 'in')
plt.ylim(y1, y2)

# plot the IR profile
data1 = np.loadtxt("resp1st.w1", dtype = float)
data2 = np.loadtxt("resp1st_im.w", dtype = float)

gnorm = gauss(1172.2 * cm_to_au, 1172.2 * cm_to_au, (30.83 * cm_to_au)**2)
gxaxis = [data1[i] / cm_to_au for i in range(0, len(data1), 5)]
line_r2, = plt.plot(gxaxis, [100 / gnorm * gauss(gxaxis[i] * cm_to_au, 1172.2 * cm_to_au, (30.83 * cm_to_au)**2) for i in range(len(gxaxis))], 'o-', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = 'red')

line_r, = ax2.plot(data1 / cm_to_au, 100 * data2 / np.max(data2), "-", linewidth = lw, color = 'r')
ax2.fill_between(data1 / cm_to_au, 100 * data2 / np.max(data2),  [0.0] * len(data1), color = 'r', alpha = .3)

# name of x, y axis and the panel
ax.set_xlabel(r'$\omega_\mathrm{c}~ (\mathrm{cm}^{-1})$', size = txtsize)
font_legend = {'family':'Times New Roman', 'weight': 'roman', 'size': 18}

ax.legend([line_l, line_l2, line_r, line_r2], [r'HEOM rates', r'FGR rates', r'IR spectra', r'Lorentzian'], loc = 'upper left', frameon = False, prop = font_legend) #, bbox_to_anchor = (0.0, 0.8)
plt.legend(title = '(a)', bbox_to_anchor = (legend_x, legend_y), frameon = False, title_fontsize = legendsize) # , loc = 'upper right'

# ==============================================================================================
#                                      Fig 2c: varying tc    
# ==============================================================================================

plt.subplot(1,100,(39,66))

tauc_collection = [1.5e5, 10000, 2000, 1000, 500, 200, 100, 50, 20, 10, 1]
rate_collection_1 = []
rate_collection_2 = []
rate_collection_3 = []

rate_0 = 1.267242438468303195e-07

rate_collection_1.append(1.001 * rate_0)
rate_collection_2.append(1.001 * rate_0)
rate_collection_3.append(1.001 * rate_0)

data1 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=10000.txt", dtype=float)
data2 = np.loadtxt("./PSD_es=0.1_ec=0.025_tauc=10000.txt", dtype=float)
data3 = np.loadtxt("./PSD_es=0.1_ec=0.0125_tauc=10000.txt", dtype=float)
rate_collection_1.append(data1[10, 1])
rate_collection_2.append(data2[1])
rate_collection_3.append(data3[1])

data1 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=2000.txt", dtype=float)
data2 = np.loadtxt("./PSD_es=0.1_ec=0.025_tauc=2000.txt", dtype=float)
data3 = np.loadtxt("./PSD_es=0.1_ec=0.0125_tauc=2000.txt", dtype=float)
rate_collection_1.append(data1[10, 1])
rate_collection_2.append(data2[1])
rate_collection_3.append(data3[1])

data1 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=1000.txt", dtype=float)
data2 = np.loadtxt("./PSD_es=0.1_ec=0.025_tauc=1000.txt", dtype=float)
data3 = np.loadtxt("./PSD_es=0.1_ec=0.0125_tauc=1000.txt", dtype=float)
rate_collection_1.append(data1[10, 1])
rate_collection_2.append(data2[1])
rate_collection_3.append(data3[1])

data1 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=500.txt", dtype=float)
data2 = np.loadtxt("./PSD_es=0.1_ec=0.025_tauc=500.txt", dtype=float)
data3 = np.loadtxt("./PSD_es=0.1_ec=0.0125_tauc=500.txt", dtype=float)
rate_collection_1.append(data1[10, 1])
rate_collection_2.append(data2[1])
rate_collection_3.append(data3[1])

data1 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=200.txt", dtype=float)
data2 = np.loadtxt("./PSD_es=0.1_ec=0.025_tauc=200.txt", dtype=float)
data3 = np.loadtxt("./PSD_es=0.1_ec=0.0125_tauc=200.txt", dtype=float)
rate_collection_1.append(data1[10, 1])
rate_collection_2.append(data2[1])
rate_collection_3.append(data3[1])

data1 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=100.txt", dtype=float)
data2 = np.loadtxt("./PSD_es=0.1_ec=0.025_tauc=100.txt", dtype=float)
data3 = np.loadtxt("./PSD_es=0.1_ec=0.0125_tauc=100.txt", dtype=float)
rate_collection_1.append(data1[10, 1])
rate_collection_2.append(data2[1])
rate_collection_3.append(data3[1])

data1 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=50.txt", dtype=float)
data2 = np.loadtxt("./PSD_es=0.1_ec=0.025_tauc=50.txt", dtype=float)
data3 = np.loadtxt("./PSD_es=0.1_ec=0.0125_tauc=50.txt", dtype=float)
rate_collection_1.append(data1[10, 1])
rate_collection_2.append(data2[1])
rate_collection_3.append(data3[1])

data1 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=20.txt", dtype=float)
data2 = np.loadtxt("./PSD_es=0.1_ec=0.025_tauc=20.txt", dtype=float)
data3 = np.loadtxt("./PSD_es=0.1_ec=0.0125_tauc=20.txt", dtype=float)
rate_collection_1.append(data1[10, 1])
rate_collection_2.append(data2[1])
rate_collection_3.append(data3[1])

data1 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=10.txt", dtype=float)
data2 = np.loadtxt("./PSD_es=0.1_ec=0.025_tauc=10.txt", dtype=float)
data3 = np.loadtxt("./PSD_es=0.1_ec=0.0125_tauc=10.txt", dtype=float)
rate_collection_1.append(data1[10, 1])
rate_collection_2.append(data2[1])
rate_collection_3.append(data3[1])

data1 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=1.txt", dtype=float)
data2 = np.loadtxt("./PSD_es=0.1_ec=0.025_tauc=1.txt", dtype=float)
data3 = np.loadtxt("./PSD_es=0.1_ec=0.0125_tauc=1.txt", dtype=float)
rate_collection_1.append(data1[10, 1])
rate_collection_2.append(data2[1])
rate_collection_3.append(data3[1])

data = np.zeros((len(tauc_collection), 4), dtype=float)
data[:, 0] = tauc_collection
data[:, 1] = rate_collection_1
data[:, 2] = rate_collection_2
data[:, 3] = rate_collection_3

k_0 = 1.267393751572242088e-07
plt.semilogx(data[:, 0], data[:, 3] / k_0, 'o-', markersize = 10, linewidth = 3, markerfacecolor = 'red', color = 'red', label = '$\Omega_\mathrm{R} =$ 6.27 cm$^{-1}$')
plt.semilogx(data[:, 0], data[:, 2] / k_0, 'o-', markersize = 10, linewidth = 3, markerfacecolor = 'greenyellow', color = 'greenyellow', label = '12.54 cm$^{-1}$')
plt.semilogx(data[:, 0], data[:, 1] / k_0, 'o-', markersize = 10, linewidth = 3, markerfacecolor = 'navy', color = 'navy', label = '25.09 cm$^{-1}$')

k_min = 1.0

tau_0 = np.linspace(1, 1e5, 10000)
plt.plot(tau_0, [k_min] * len(tau_0), '--', linewidth = lw, color = 'black')
plt.text(2000, 0.97, r'lossless limit', color = 'black', size = 24)
plt.text(2, 0.97, r'lossy limit', color = 'black', size = 24)

y_major_locator = MultipleLocator(0.1)
y_minor_locator = MultipleLocator(0.05)

# # lower x-axis and LHS y-axis
ax = plt.gca()
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, pad = 10)
ax.tick_params(which = 'minor', length = 5)

ax.vlines([196.05], 0.95, 1.4, linestyles = 'dashed', colors = ['navy'], lw = 3.0) 
ax.vlines([2 * 196.05], 0.95, 1.2, linestyles = 'dashed', colors = ['greenyellow'], lw = 3.0) 
ax.vlines([4 * 196.05], 0.95, 1.1, linestyles = 'dashed', colors = ['red'], lw = 3.0) 

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(which = 'both', direction = 'in', labelsize = 30)
plt.xlim(1, 1e5)
plt.ylim(0.95, 1.4)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 15)
ax2.tick_params(which = 'minor', length = 5)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(0.9, 1.4)

ax.set_xlabel(r'$\tau_\mathrm{c}\ (\mathrm{fs})$', size = 32)
ax.set_ylabel(r'$k / k_0$', size = 32)
ax.legend(loc = 'upper left', frameon = False, prop = font_legend)
plt.legend(title = '(b)', bbox_to_anchor = (legend_x, legend_y), frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                       Fig 2a: ec effect 
# ==============================================================================================

plt.subplot(1,100,(75,100))

data = np.loadtxt("etac_1155.txt", dtype=float)

# ==================== plotting ddG ==========================
def ddg(x):
    return - (300 / au_to_K) * np.log(x)

def kvsc_2(x):
    rescaling = 0.5
    Omega_R = 2 * mu_0 * wc * x
    tau_c = 1000 * fs_to_au
    return Omega_R**2 / np.sqrt(sigma_2) / (2 + Omega_R**2 * tau_c / np.sqrt(sigma_2)) * Bose(w0) * rescaling

x_major_locator = MultipleLocator(20)
x_minor_locator = MultipleLocator(10)
y_major_locator = MultipleLocator(0.1)
y_minor_locator = MultipleLocator(0.05)

ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = lmajortick, labelsize = 30, pad = 10)
ax.tick_params(which = 'minor', length = lminortick)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

y1, y2 = 0.95, 1.45
plt.tick_params(labelsize = lsize, which = 'both', direction = 'in')
plt.xlim(0.0, 110.0)
plt.ylim(y1, y2)

ax.set_xlabel(r'$\Omega_\mathrm{R}$ (cm$^{-1}$)', size = txtsize)
ax.set_ylabel(r'$k / k_0$      ', size = txtsize, labelpad = 15)

# RHS y-axis
y_major_locator = MultipleLocator(0.1)
y_minor_locator = MultipleLocator(0.05)

ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = lmajortick, labelsize = 30)
ax2.tick_params(which = 'minor', length = lminortick)

y2_label = ax2.get_yticklabels()
[y2_label_temp.set_fontname('Times New Roman') for y2_label_temp in y2_label]

# for plotting range
x = np.linspace(0.00, 0.22, 10000)

# k / k_0 results: HEOM vs FGR
line1, = ax.plot(2 * mu_0 * wc / cm_to_au * data[:, 0], data[:, 1] / k_0, "o", markersize = 10, markerfacecolor = 'white', color = 'navy')
line2, = ax.plot(2 * mu_0 * wc / cm_to_au * x, 1 + kvsc_2(x) / k_0, linewidth = lw, color = 'navy')

# ddG results: HEOM vs FGR
line3, = ax2.plot(2 * mu_0 * wc / cm_to_au * data[:, 0], ddg(data[:, 1] / k_0) / kcal_to_au, "o", linewidth = lw * 0.75, markersize = 10, markerfacecolor = 'white', color = 'r')
line4, = ax2.plot(2 * mu_0 * wc / cm_to_au * x, ddg(1 + kvsc_2(x) / k_0) / kcal_to_au, linewidth = lw, color = 'r')

x_00, x_01 = 12, 2
x_10, x_11 = 90, 106
y_0 = 1.135
y_1 = 1.075
arrowrange_0 = np.linspace(x_00, x_01, 1000)
arrowrange_1 = np.linspace(x_10, x_11, 1000)
plt.plot(arrowrange_0, [y_0] * len(arrowrange_0), lw = 2.5, color = 'navy')
plt.plot(arrowrange_1, [y_1] * len(arrowrange_1), lw = 2.5, color = 'red')
arrowleft(ax, arrowrange_0 * 0.98, [y_0] * len(arrowrange_0), style = 'to', arrow_size = 2, arrow_style = 'full', arrow_angle = 40, color = 'navy')
arrowright(ax, arrowrange_1 * 1.02, [y_1] * len(arrowrange_1), style = 'to', arrow_size = 2, arrow_style = 'full', arrow_angle = 40, color = 'red')

plt.tick_params(labelsize = 0, which = 'both', direction = 'in')
plt.ylim(y1, y2)

y1, y2 = -0.2, 0.05     # y-axis range: (y1, y2)
plt.tick_params(labelsize = lsize, which = 'both', direction = 'in')
plt.ylim(y1, y2)

ax2.set_ylabel(r'$\Delta(\Delta G^{\ddag})\ (\mathrm{Kcal/mol})$', size = txtsize)

ax.legend([line1, line2, line3, line4], [r"$k / k_0$ (HEOM)", r"$k / k_0$ (FGR)", r"$\Delta(\Delta G^{\ddag})$ (HEOM)", r"$\Delta(\Delta G^{\ddag})$ (FGR)"], loc = 'upper center', frameon = False, prop = font_legend, ncol = 2) #, bbox_to_anchor = (0.0, 0.4), 
plt.legend(title = '(c)', bbox_to_anchor = (legend_x, legend_y), frameon = False, title_fontsize = legendsize)



plt.savefig("Fig_3.pdf", bbox_inches='tight')
