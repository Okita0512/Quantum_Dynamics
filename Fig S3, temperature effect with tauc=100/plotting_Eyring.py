import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from matplotlib.pyplot import MultipleLocator, tick_params
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

# ================= global ====================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# ================= data reading ====================   

Temp = np.array([280, 290, 300, 310, 320])

curve0 = []     # outside
curve1 = []     # 0.00625    
curve2 = []     # 0.0125    
curve3 = []     # 0.025    
curve4 = []     # 0.05     
curve5 = []     # 0.1

count = 0
for t in Temp:

    data1 = np.loadtxt("./PSD_es=0.1_ec=0.00625_tauc=100_temp=%d.txt" % (Temp[count]), dtype = float)
    data2 = np.loadtxt("./PSD_es=0.1_ec=0.0125_tauc=100_temp=%d.txt" % (Temp[count]), dtype = float)
    data3 = np.loadtxt("./PSD_es=0.1_ec=0.025_tauc=100_temp=%d.txt" % (Temp[count]), dtype = float)
    data4 = np.loadtxt("./PSD_es=0.1_ec=0.05_tauc=100_temp=%d.txt" % (Temp[count]), dtype = float)
    data5 = np.loadtxt("./PSD_es=0.1_ec=0.1_tauc=100_temp=%d.txt" % (Temp[count]), dtype = float)
    curve0.append(np.log(data1[0, 1] * fs_to_au  / (t / au_to_K)))
    curve1.append(np.log(data1[10, 1] * fs_to_au / (t / au_to_K)))
    curve2.append(np.log(data2[10, 1] * fs_to_au / (t / au_to_K)))
    curve3.append(np.log(data3[10, 1] * fs_to_au / (t / au_to_K)))
    curve4.append(np.log(data4[10, 1] * fs_to_au / (t / au_to_K)))
    curve5.append(np.log(data5[10, 1] * fs_to_au / (t / au_to_K)))

    count += 1

z0 = np.polyfit(1. / Temp, curve0, 1)
z1 = np.polyfit(1. / Temp, curve1, 1)
z2 = np.polyfit(1. / Temp, curve2, 1)
z3 = np.polyfit(1. / Temp, curve3, 1)
z4 = np.polyfit(1. / Temp, curve4, 1)
z5 = np.polyfit(1. / Temp, curve5, 1)

x = np.linspace(1 / 400, 1 / 200, 1000)
y_fit_0 = np.polyval(z0, x)
y_fit_1 = np.polyval(z1, x)
y_fit_2 = np.polyval(z2, x)
y_fit_3 = np.polyval(z3, x)
y_fit_4 = np.polyval(z4, x)
y_fit_5 = np.polyval(z5, x)

# =================================================

ec = np.array([0, 0.00625, 0.0125, 0.025, 0.05, 0.1])
dH_out = [- z0[0] / (au_to_K * kcal_to_au)] * len(ec)
dH_in = []
dS_out = [z0[1] / (kcal_to_au * 1000)] * len(ec)
dS_in = []

dH_in.append(- z0[0] / (au_to_K * kcal_to_au))
dS_in.append(z0[1] / (kcal_to_au * 1000))
dH_in.append(- z1[0] / (au_to_K * kcal_to_au))
dS_in.append(z1[1] / (kcal_to_au * 1000))
dH_in.append(- z2[0] / (au_to_K * kcal_to_au))
dS_in.append(z2[1] / (kcal_to_au * 1000))
dH_in.append(- z3[0] / (au_to_K * kcal_to_au))
dS_in.append(z3[1] / (kcal_to_au * 1000))
dH_in.append(- z4[0] / (au_to_K * kcal_to_au))
dS_in.append(z4[1] / (kcal_to_au * 1000))
dH_in.append(- z5[0] / (au_to_K * kcal_to_au))
dS_in.append(z5[1] / (kcal_to_au * 1000))

print('outside coefficients:', - z0[0] / (au_to_K * cm_to_au), z0[1] / (cm_to_au), r'$cm^{-1}$')
print('0.00625 coefficients:', - z1[0] / (au_to_K * cm_to_au), z1[1] / (cm_to_au), r'$cm^{-1}$')
print('0.0125 coefficients:', - z2[0] / (au_to_K * cm_to_au), z2[1] / (cm_to_au), r'$cm^{-1}$')
print('0.025 coefficients:', - z3[0] / (au_to_K * cm_to_au), z3[1] / (cm_to_au), r'$cm^{-1}$')
print('0.05 coefficients:', - z4[0] / (au_to_K * cm_to_au), z4[1] / (cm_to_au), r'$cm^{-1}$')
print('0.1 coefficients:', - z5[0] / (au_to_K * cm_to_au), z5[1] / (cm_to_au), r'$cm^{-1}$')

# ==============================================================================================
rescaling = 0.5
factor = 1.0

ec_plot = np.linspace(1e-5, 0.11, 1000)
tauc = 100 * fs_to_au
mu = 0.214
w0 = 1172.2 * cm_to_au
wc = w0
gamma_c = 1e10 * cm_to_au
k0 = 1.262736060279871165e-07
T0 = 300 / au_to_K
beta = 1. / T0

def coth(x):                                # mathematical function, cot(x)
    return 1 / np.tanh(x)

def Drude(x):                               # the molecular bath spectral density function, J_v(w)
    lam = 83.7 * cm_to_au
    gam = 200 * cm_to_au
    return (2 * lam * gam * x / (x**2 + gam**2)) * coth(beta * x / 2)

def λc(tau_c, ωc, gamma):                     # function to calculate loss bath reorganization energy from cavity lifetime
    return (1.0 - np.exp(- beta * ωc)) * (gamma**2 + ωc**2) / (2 * tau_c * gamma)

def gen_jw(w, omega_c, eta_c, lam, gamma):  # the effective spectral density function

    J0 = (2 * lam * gamma * w / (w**2 + gamma**2)) # * (2 * omega_c)     # secondary bath dissipation operator: np.sqrt(2 w_c) q_c, which is just (a + a^+)
    zeta = np.sqrt(2 / omega_c) * eta_c

    return ((omega_c**4 * zeta**2 * J0) / ((omega_c**2 - w**2 + (w * J0 / gamma))**2 + (J0)**2))

def gauss(x, x0, sigma_2):                  # gaussian distribution, with center x0 and variance sigma_2
    return (1./np.pi) * np.sqrt(sigma_2) / ((x - x0)**2 + sigma_2)
#    return (1 / np.sqrt(2 * np.pi * sigma_2)) * np.exp(- (x - x0)**2 / (2 * sigma_2))

# to get the variance
Rij = 0.231
wi = np.linspace(1e-10, 200 * cm_to_au, 10000000)     # for intergration. Better to be larger (at least 10^3)
y = Drude(wi)
sigma_2 = integrate.trapz(y, wi)
# sigma_2 = (0.01 * cm_to_au)**2 # 
sigma_2 = Rij**2 * sigma_2 / (np.pi)
print("sigma_2 = ", np.sqrt(sigma_2) / cm_to_au, '\t cm^-1')

wc_scan_2 = np.linspace(0.8 * wc, 1.2 * wc, 1000)        # for intergration, Better to be larger (at least 10^5)
lambda_c = λc(tauc, wc, gamma_c)          # loss bath reorganization energy

kvsc = np.zeros((len(ec_plot)), dtype = float)
count = 0
for etac in ec_plot:
    def intergrant(x):
        return 2 * mu**2 * gen_jw(x, wc, etac, lambda_c, gamma_c) * gauss(x, w0, sigma_2) * np.exp(- w0 / T0)
    y = intergrant(wc_scan_2)
    result = integrate.trapz(y, wc_scan_2)
    kvsc[count] = result * rescaling
    count += 1

# kvsc = 4 * ec_plot**2 * mu**2 * wc**2 * tauc * np.exp(- w0 / T0)

# ==============================================================================================

lw = 3.0
# size for legend
legendsize = 48         
font_legend = {'family':'Times New Roman', 'weight': 'roman', 'size': 20}
# axis label size
lsize = 30             
txtsize = 32
# tick length
lmajortick = 15
lminortick = 5
legend_x, legend_y = - 0.125, 1.03

unitlen = 9
fig = plt.figure(figsize=(3.3 * unitlen, 0.9 * unitlen), dpi = 128)
fig.subplots_adjust(wspace = 0.3)

# ==============================================================================================
#                                         Panel A     
# ==============================================================================================

plt.subplot(1,3,1)

plt.plot(1. / Temp, [curve0[i] - curve0[i] for i in range(len(curve0))], 'o', markersize = 10, label = "Outside Cavity", color = 'k')
plt.plot(1. / Temp, [curve1[i] - curve0[i] for i in range(len(curve0))], 'o', markersize = 10, label = r"$\Omega_\mathrm{R} =$ 3.14 cm$^{-1}$", color = 'skyblue')
plt.plot(1. / Temp, [curve2[i] - curve0[i] for i in range(len(curve0))], 'o', markersize = 10, label = r"$\Omega_\mathrm{R} =$ 6.27 cm$^{-1}$", color = 'violet')
plt.plot(1. / Temp, [curve3[i] - curve0[i] for i in range(len(curve0))], 'o', markersize = 10, label = r"$\Omega_\mathrm{R} =$ 12.54 cm$^{-1}$", color = 'g')
plt.plot(1. / Temp, [curve4[i] - curve0[i] for i in range(len(curve0))], 'o', markersize = 10, label = r"$\Omega_\mathrm{R} =$ 25.09 cm$^{-1}$", color = 'orange')
plt.plot(1. / Temp, [curve5[i] - curve0[i] for i in range(len(curve0))], 'o', markersize = 10, label = r"$\Omega_\mathrm{R} =$ 50.17 cm$^{-1}$", color = 'red')

plt.plot(x, y_fit_0 - y_fit_0, '--', linewidth = 2, color = 'k')
plt.plot(x, y_fit_1 - y_fit_0, '-', linewidth = 2, color = 'skyblue')
plt.plot(x, y_fit_2 - y_fit_0, '-', linewidth = 2, color = 'violet')
plt.plot(x, y_fit_3 - y_fit_0, '-', linewidth = 2, color = 'g')
plt.plot(x, y_fit_4 - y_fit_0, '-', linewidth = 2, color = 'orange')
plt.plot(x, y_fit_5 - y_fit_0, '-', linewidth = 2, color = 'red')

# x and y range of plotting 
x1, x2 = 0.00305, 0.00365
y1, y2 = -0.1, 1.2     #-15, -10 # y-axis range: (y1, y2)

# scale for major and minor locator
x_major_locator = MultipleLocator(0.0002)
x_minor_locator = MultipleLocator(0.0001)
y_major_locator = MultipleLocator(0.2)
y_minor_locator = MultipleLocator(0.1)

# x-axis and LHS y-axis
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

ax.set_xlabel(r'$1 / T ~ (\mathrm{K}^{-1})$', size = txtsize, labelpad = 10)
ax.set_ylabel(r'$\ln (k / k_0)$', size = txtsize, labelpad = 20)
ax.legend(frameon = False, loc = 'upper center', prop = font_legend, ncol = 2) # , markerscale = 1.5
plt.legend(title = '(a)', bbox_to_anchor = (legend_x - 0.05, legend_y), frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                         Panel B     
# ==============================================================================================

plt.subplot(1,3,2)

font_legend = {'family':'Times New Roman', 'weight': 'roman', 'size': 28}

dH0 = - z0[0] / (au_to_K * kcal_to_au)
ddH = ((w0 - T0) - dH0 * kcal_to_au) / (1 + k0 / kvsc)

# plt.plot(ec, , '--', linewidth = 5, color = 'black', label = 'Outside Cavity')
plt.plot(2 * w0 * mu / cm_to_au * ec, [dH_in[i] - dH_out[i] for i in range(len(dH_in))], 'o-', markersize = 15, markerfacecolor = 'white', color = 'navy', label = 'HEOM')
plt.plot(2 * w0 * mu / cm_to_au * ec_plot, factor * ddH / kcal_to_au, '--', linewidth = 5, color = 'goldenrod', label = 'FGR')

x_major_locator = MultipleLocator(10)
x_minor_locator = MultipleLocator(2)
y_major_locator = MultipleLocator(0.5)
y_minor_locator = MultipleLocator(0.1)

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

y1, y2 = -0.2, 1.8
plt.tick_params(labelsize = lsize, which = 'both', direction = 'in')
plt.xlim(0.0, 54.0)
plt.ylim(y1, y2)

ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = lmajortick, labelsize = 30)
ax2.tick_params(which = 'minor', length = lminortick)

y2_label = ax2.get_yticklabels()
[y2_label_temp.set_fontname('Times New Roman') for y2_label_temp in y2_label]

plt.tick_params(labelsize = 0, which = 'both', direction = 'in')
plt.ylim(y1, y2)

ax.set_xlabel(r'$\Omega_\mathrm{R}$ (cm$^{-1}$)', size = txtsize)
ax.set_ylabel(r'$\Delta \Delta H^{\ddag}\ (\mathrm{KCal \cdot mol^{-1}})$', size = txtsize, labelpad = 20)
ax.legend(frameon = False, loc = 'upper left', prop = font_legend, markerscale = 1)
plt.legend(title = '(b)', bbox_to_anchor = (legend_x, legend_y), frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                      Panel C 
# ==============================================================================================

plt.subplot(1,3,3)

# dS0 = z0[1] / (au_to_K * kcal_to_au) * 100
# ddS = (ddH / T0 + np.log(1 + kvsc / k0)) / (au_to_K * kcal_to_au) * 100

dS0 = z0[1] / (kcal_to_au * 1000)
ddS = (ddH / T0 + np.log(1 + kvsc / k0)) / (kcal_to_au * 1000)

# plt.plot(ec, dS_out, '--', linewidth = 5, color = 'black', label = 'Outside Cavity')
plt.plot(2 * w0 * mu / cm_to_au * ec, [dS_in[i] - dS_out[i] for i in range(len(dS_in))], 'o-', markersize = 15, markerfacecolor = 'white', color = 'navy', label = 'HEOM')
plt.plot(2 * w0 * mu / cm_to_au * ec_plot, factor * ddS, '--', linewidth = 5, color = 'goldenrod', label = 'FGR')

x_major_locator = MultipleLocator(10)
x_minor_locator = MultipleLocator(2)
y_major_locator = MultipleLocator(0.5)
y_minor_locator = MultipleLocator(0.1)

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

y1, y2 = -0.2, 2.5
plt.tick_params(labelsize = lsize, which = 'both', direction = 'in')
plt.xlim(0.0, 54.0)
plt.ylim(y1, y2)

ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = lmajortick, labelsize = 30)
ax2.tick_params(which = 'minor', length = lminortick)

y2_label = ax2.get_yticklabels()
[y2_label_temp.set_fontname('Times New Roman') for y2_label_temp in y2_label]

plt.tick_params(labelsize = 0, which = 'both', direction = 'in')
plt.ylim(y1, y2)

ax.set_xlabel(r'$\Omega_\mathrm{R}$ (cm$^{-1}$)', size = txtsize)
ax.set_ylabel(r'$\Delta \Delta S^{\ddag}\ (\mathrm{Cal \cdot mol^{-1} \cdot K^{-1}})$      ', size = txtsize, labelpad = 15)
ax.legend(frameon = False, loc = 'upper left', prop = font_legend, markerscale = 1)
plt.legend(title = '(c)', bbox_to_anchor = (legend_x, legend_y), frameon = False, title_fontsize = legendsize)





plt.savefig("Fig_temp_Eyring_100.pdf", bbox_inches='tight')