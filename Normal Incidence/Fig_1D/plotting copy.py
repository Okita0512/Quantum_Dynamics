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
unitlen = 6
legendsize = 48         # size for legend
font_legend = {'family':'Times New Roman',
        'style':'normal', # 'italic'
        'weight':'normal', #or 'blod'
        'size':28
        }

fig, ax = plt.subplots(2, 3, figsize=(2.7 * unitlen, 2.0 * unitlen), dpi = 512, sharey = 'row')
fig.subplots_adjust(hspace = 0.25, wspace = 0.0)

# ==============================================================================================
#                                       Global Parameters     
# ==============================================================================================
conv = 27.211397                            # 1 a.u. = 27.211397 eV
fs_to_au = 41.341                           # 1 fs = 41.341 a.u.
cm_to_au = 4.556335e-06                     # 1 cm^-1 = 4.556335e-06 a.u.
au_to_K = 3.1577464e+05                     # 1 au = 3.1577464e+05 K
kcal_to_au = 1.5936e-03                     # 1 kcal/mol = 1.5936e-3 a.u.

# ===== Cavity parameters =====
w0 = 1190 * cm_to_au

# ===== Auxiliary functions =====
def cot(x):
    return np.cos(x) / np.sin(x)

def omega_k(theta, wc):
    return wc * np.sqrt(1 + np.tan(theta)**2)

def brownian(w, omega_c, alpha, theta):  # the effective spectral density function
    return ((2 * alpha * w) / ((omega_c**2 * (1 + np.tan(theta)**2) - w**2)**2 + (alpha * w)**2))

# ==============================================================================================
#                                       Plotting Fig 1a     
# ==============================================================================================
plt.subplot(2,3,1)

wc = w0
a_ratio = 360 / (2 * np.pi)
theta_scan = np.linspace(- np.pi / 3, np.pi / 3, 1000)

plt.plot(theta_scan * a_ratio, omega_k(theta_scan, wc) / cm_to_au, '--', linewidth = lw, color = 'red', label = r'$\omega_{\mathbf{k}}$')
plt.plot(theta_scan * a_ratio, [w0 / cm_to_au] * len(theta_scan), '--', linewidth = lw, color = 'black', label = r'$\omega_0$')

# scale for major and minor locator
x_major_locator = MultipleLocator(20)
x_minor_locator = MultipleLocator(10)
y_major_locator = MultipleLocator(400)
y_minor_locator = MultipleLocator(80)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 30)
ax.tick_params(which = 'minor', length = 5)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

y1, y2 = 600, 2000
plt.tick_params(labelsize = 30, which = 'both', direction = 'in')
plt.xlim(- 55, 55)
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
ax.set_xlabel(r'Incident Angle $\theta~ (^\circ)$', size = axis_size)
ax.set_ylabel(r'$\omega\ (\mathrm{cm}^{-1})$', size = axis_size)
ax.legend(loc = 'upper center', frameon = False, prop = font_legend)
plt.legend(title = '(a)', frameon = False, title_fontsize = legendsize)

# ==============================================================================================
#                                       Plotting Fig 1b     
# ==============================================================================================

plt.subplot(2,3,2)

beta = au_to_K / 300

theta_max = np.pi / 2 - 1e-6
def DOS(w):
    return w * np.exp(- beta * w) * np.sqrt(w**2 - wc**2) # / (wc * np.tan(theta_max) * np.sqrt(w**2 - wc**2))

w_plot = np.linspace(wc + 0.01 * cm_to_au, 2000 * cm_to_au, 1000000)

plt.plot(DOS(w_plot) * 1000, w_plot / cm_to_au, '-', linewidth = lw, color = 'blue', label = r'DOS')
plt.plot(np.linspace(-1, 100, 1000), [w0 / cm_to_au] * 1000, '--', linewidth = lw, color = 'black')

print(max(DOS(w_plot)))

# x-axis and LHS y-axis
x_major_locator = MultipleLocator(10)
x_minor_locator = MultipleLocator(5)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 15, labelsize = 30)
ax.tick_params(which = 'minor', length = 5)

plt.axvline(0.0, y1, y2, linewidth = lw, color = 'black')

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(labelsize = 30, which = 'both', direction = 'in')
plt.xlim(0.0, 0.00005)
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
ax.set_xlabel(r'Intensity ($\times 10^{-3}~ \mathrm{a.u.}$)', size = axis_size)
# ax.set_ylabel(r'$\omega\ (\mathrm{cm}^{-1})$', size = 36)
ax.legend(loc = 'upper center', frameon = False, prop = font_legend)
plt.legend(title = '(b)', frameon = False, title_fontsize = legendsize)




plt.savefig("figure_DOS_2.pdf", bbox_inches='tight')