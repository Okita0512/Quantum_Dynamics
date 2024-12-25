import matplotlib.pyplot as plt
import numpy as np
fig, ax = plt.subplots()
from scipy import integrate
from matplotlib.pyplot import MultipleLocator, tick_params

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
lsize = 30             
txtsize = 24
# tick length
lmajortick = 15
lminortick = 5


# unitlen = 7
# fig = plt.figure(figsize=(3.3 * unitlen, 0.85 * unitlen), dpi = 512)

# ==============================================================================================
#                                      Fig a: Ahrennius plot  
# ==============================================================================================
alpha_c = 60 * cm_to_au
w0 = 1172 * cm_to_au

def cot(x):
    return np.cos(x) / np.sin(x)

def brownian(w, omega_c, alpha, theta):  # the effective spectral density function
    return ((2 * alpha * w) / ((omega_c**2 * (1 + np.tan(theta)**2) - w**2)**2 + (alpha * w)**2))

# ============================================================================
# plot the integrant
# ============================================================================

# angle = np.linspace(0.00001, np.pi / 3, 1000)
# def integrant(theta):
#          jw = brownian(w0, w0, alpha_c, theta) * np.sqrt(1 + cot(theta)**2) / np.cos(theta)**4
#          return jw
# 
# J_angle = integrant(angle)
# 
# plt.plot(angle * 360 / (2 * np.pi), J_angle, '-')

# ============================================================================
# scan wc
# ============================================================================

w_scan = np.linspace(0.5 * w0, 1.5 * w0, 100)
angle = np.linspace(0.00001, np.pi / 3, 1000000)
spd = []
for z in w_scan:
    def integrant(theta):
        jw = brownian(z, w0, alpha_c, theta) * np.sqrt(1 + cot(theta)**2) / np.cos(theta)**4
        return jw
    y = integrant(angle)
    Jz = integrate.trapz(y, angle)
    spd.append(Jz)

plt.plot(w_scan / cm_to_au, spd, '-')

# ============================================================================
# scan theta
# ============================================================================

# phi_scan = np.linspace(0.00001, np.pi / 3, 100)
# angle = np.linspace(0.00001, np.pi / 3, 1000000)
# spd = []
# for z in phi_scan:
#     wc = w0 * np.cos(z)
#     def integrant(theta):
#         jw = brownian(wc, w0, alpha_c, theta) * np.sqrt(1 + cot(theta)**2) / np.cos(theta)**4
#         return jw
#     y = integrant(angle)
#     Jz = integrate.trapz(y, angle)
#     spd.append(Jz)
# 
# plt.plot(phi_scan * 360 / (2 * np.pi), spd, '-')

# ============================================================================

# x_major_locator = MultipleLocator(0.005)
# x_minor_locator = MultipleLocator(0.001)
# y_major_locator = MultipleLocator(0.2)
# y_minor_locator = MultipleLocator(0.1)
# 
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# ax.xaxis.set_minor_locator(x_minor_locator)
# ax.yaxis.set_major_locator(y_major_locator)
# ax.yaxis.set_minor_locator(y_minor_locator)
# ax.tick_params(which = 'major', length = lmajortick, labelsize = 10, pad = 10)
# ax.tick_params(which = 'minor', length = lminortick)
# 
# x1_label = ax.get_xticklabels()
# [x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
# y1_label = ax.get_yticklabels()
# [y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]
# 
# y1, y2 = 0, 0.7
# plt.tick_params(labelsize = lsize, which = 'both', direction = 'in')
# plt.xlim(0, 0.011)
# plt.ylim(y1, y2)
# 
# ax2 = ax.twinx()
# ax2.yaxis.set_major_locator(y_major_locator)
# ax2.yaxis.set_minor_locator(y_minor_locator)
# ax2.tick_params(which = 'major', length = lmajortick, labelsize = 10)
# ax2.tick_params(which = 'minor', length = lminortick)
# 
# y2_label = ax2.get_yticklabels()
# [y2_label_temp.set_fontname('Times New Roman') for y2_label_temp in y2_label]
# 
# plt.tick_params(labelsize = 0, which = 'both', direction = 'in')
# plt.ylim(y1, y2)
# 
# ax.set_xlabel(r'$\eta_\mathrm{c}$', size = txtsize)
# ax.set_ylabel(r'$\Delta H^{\ddag}\ (\mathrm{Kcal/mol})$', size = txtsize)
# ax.legend(frameon = False, loc = 'center right', prop = font_legend, markerscale = 1)
# plt.legend(title = '(a)', frameon = False, title_fontsize = legendsize)


plt.show()

# plt.savefig("figure_Jeff.pdf", bbox_inches='tight')