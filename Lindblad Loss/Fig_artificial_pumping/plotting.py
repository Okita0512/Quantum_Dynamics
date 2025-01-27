import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
from matplotlib.pyplot import MultipleLocator, tick_params
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams["font.family"] = "Helvetica"

# ================= global ====================

conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

lw = 3.0
# size for legend
legendsize = 48         
font_legend = {'family':'Times New Roman', 'weight': 'roman', 'size': 24}
# axis label size
lsize = 25            
txtsize = 24
# tick length
lmajortick = 15
lminortick = 5

unitlen = 7
fig = plt.figure(figsize=(1.0 * unitlen, 0.6 * unitlen), dpi = 128)

# ==============================================================================================
#                                      Auxiliary functions  
# ==============================================================================================
def cot(x):
    return np.cos(x) / np.sin(x)

def Brownian(w, omega_c, eta_c, Gamma):
    lam = eta_c**2 * omega_c
    return 2 * lam * omega_c**2 * Gamma * w / ((w**2 - omega_c**2)**2 + (Gamma * w)**2) * np.heaviside(w, 0)

def Lorentzian(w, omega_c, eta_c, Gamma):
    gc = eta_c * omega_c
    return (gc**2) * Gamma / ((w - omega_c)**2 + Gamma**2) # / np.pi

# ============================================================================
# scan wc
# ============================================================================

wc = 1172 * cm_to_au
eta_c = 0.1
tauc = 200 * fs_to_au
Gamma = 1. / tauc
w_scan = np.linspace(-800 * cm_to_au, 2000 * cm_to_au, 10000)
w_fill = np.linspace(-800 * cm_to_au, 0, 2000)

plt.semilogy(w_scan / cm_to_au, Brownian(w_scan, wc, eta_c, Gamma), '-', linewidth = lw, color = 'black', label = "Brownian")
plt.semilogy(w_scan / cm_to_au, Lorentzian(w_scan, wc, eta_c, Gamma / 2), '-', linewidth = lw, color = 'deepskyblue', label = "Lorentzian")
plt.fill_between(w_fill / cm_to_au, Lorentzian(w_fill, wc, eta_c, Gamma / 2), [0.0] * len(w_fill), color = 'deepskyblue', alpha = .3)

plt.text(-650, 4e-8, "Artificial", size = 18)
plt.text(-670, 1e-8, "pumping", size = 18)

x_major_locator = MultipleLocator(500)
x_minor_locator = MultipleLocator(100)
# y_major_locator = MultipleLocator(0.2)
# y_minor_locator = MultipleLocator(0.1)
# 
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = lmajortick, labelsize = 10, pad = 10)
ax.tick_params(which = 'minor', length = lminortick)

ax.vlines([0], 0.0, 0.1, linestyles = 'dashed', colors = ['black'], lw = 3.0) 

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

# y1, y2 = 0, 0.7
plt.tick_params(labelsize = lsize, which = 'both', direction = 'in')
plt.xlim(-800, 2000)
plt.ylim(1e-9, 1e-2)
# 
# ax2 = ax.twinx()
# ax2.yaxis.set_major_locator(y_major_locator)
# ax2.yaxis.set_minor_locator(y_minor_locator)
# ax2.tick_params(which = 'major', length = lmajortick, labelsize = 10)
# ax2.tick_params(which = 'minor', length = lminortick)
# 
# y2_label = ax2.get_yticklabels()
# [y2_label_temp.set_fontname('Times New Roman') for y2_label_temp in y2_label]

# plt.tick_params(labelsize = 0, which = 'both', direction = 'in')
# plt.ylim(y1, y2)
# 
ax.set_xlabel(r'$\omega~ (\mathrm{cm}^{-1})$', size = txtsize)
ax.set_ylabel(r'Intensity (a.u.)', size = txtsize)
ax.legend(frameon = False, loc = 'lower right', prop = font_legend, markerscale = 1)
# plt.legend(title = '(a)', frameon = False, title_fontsize = legendsize)


# plt.show()

plt.savefig("Fig_Jeff.pdf", bbox_inches='tight')